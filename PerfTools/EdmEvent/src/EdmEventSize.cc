/** \file PerfTools/EdmEvent/interface/EdmEventSize.cc
 *
 *  \author Vincenzo Innocente
 */
#include "PerfTools/EdmEvent/interface/EdmEventSize.h"
#include <valarray>
#include <functional>
#include <algorithm>
#include <ostream>
#include <limits>
#include <cassert>
#include <numeric>

#include "Rtypes.h"
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TStyle.h"
#include "TObjArray.h"
#include "TBranch.h"
#include "TH1.h"
#include "TCanvas.h"
#include "Riostream.h"
#include "TDataMember.h"
#include "TLeaf.h"

#include "TBufferFile.h"

namespace {

  enum Indices { kUncompressed, kCompressed };

  typedef std::valarray<Long64_t> size_type;

  size_type getBasketSize(TBranch*);

  size_type getBasketSize(TObjArray* branches) {
    size_type result(static_cast<Long64_t>(0), 2);
    size_t n = branches->GetEntries();
    for (size_t i = 0; i < n; ++i) {
      TBranch* b = dynamic_cast<TBranch*>(branches->At(i));
      assert(b != nullptr);
      result += getBasketSize(b);
    }
    return result;
  }

  size_type getBasketSize(TBranch* b) {
    size_type result(static_cast<Long64_t>(0), 2);
    if (b != nullptr) {
      if (b->GetZipBytes() > 0) {
        result[kUncompressed] = b->GetTotBytes();
        result[kCompressed] = b->GetZipBytes();
      } else {
        result[kUncompressed] = b->GetTotalSize();
        result[kCompressed] = b->GetTotalSize();
      }
      result += getBasketSize(b->GetListOfBranches());
    }
    return result;
  }

  size_type getTotalSize(TBranch* br) {
    TBufferFile buf(TBuffer::kWrite, 10000);
    TBranch::Class()->WriteBuffer(buf, br);
    size_type size = getBasketSize(br);
    if (br->GetZipBytes() > 0)
      size[kUncompressed] += buf.Length();
    return size;
  }

}  // namespace

namespace perftools {

  EdmEventSize::EdmEventSize() : m_nEvents(0), m_mode(perftools::EdmEventSize::Mode::branches) {}

  EdmEventSize::EdmEventSize(std::string const& fileName, std::string const& treeName, Mode mode)
      : m_nEvents(0), m_mode(mode) {
    parseFile(fileName, treeName);
  }

  EdmEventSize::Leaves getLeaves(TBranch* b) {
    EdmEventSize::Leaves new_leaves{};
    auto subBranches = b->GetListOfBranches();
    const size_t nl = subBranches->GetEntries();
    if (nl == 0) {
      TLeaf* l = dynamic_cast<TLeaf*>(b->GetListOfLeaves()->At(0));
      if (l == nullptr)
        return new_leaves;

      std::string const leaf_name = l->GetName();
      std::string const leaf_type = l->GetTypeName();
      double compressed_size = l->GetBranch()->GetZipBytes();
      double uncompressed_size = l->GetBranch()->GetTotBytes();
      std::string full_name = leaf_name + '|' + leaf_type;
      full_name.erase(std::remove(full_name.begin(), full_name.end(), ' '), full_name.end());
      size_t nEvents = l->GetBranch()->GetEntries();
      new_leaves.push_back(EdmEventSize::LeafRecord(full_name, nEvents, compressed_size, uncompressed_size));
    } else {
      for (size_t j = 0; j < nl; ++j) {
        TBranch* subBranch = dynamic_cast<TBranch*>(subBranches->At(j));
        if (subBranch == nullptr)
          continue;
        auto leaves = getLeaves(subBranch);
        new_leaves.insert(new_leaves.end(), leaves.begin(), leaves.end());
      }
    }
    return new_leaves;
  }

  void EdmEventSize::parseFile(std::string const& fileName, std::string const& treeName) {
    m_fileName = fileName;
    m_branches.clear();

    TFile* file = TFile::Open(fileName.c_str());
    if (file == nullptr || (!(*file).IsOpen()))
      throw Error("unable to open data file " + fileName, 7002);

    TObject* o = file->Get(treeName.c_str());
    if (o == nullptr)
      throw Error("no object \"" + treeName + "\" found in file: " + fileName, 7003);

    TTree* events = dynamic_cast<TTree*>(o);
    if (events == nullptr)
      throw Error("object \"" + treeName + "\" is not a TTree in file: " + fileName, 7004);

    m_nEvents = events->GetEntries();
    if (m_nEvents == 0)
      throw Error("tree \"" + treeName + "\" in file " + fileName + " contains no Events", 7005);

    TObjArray* branches = events->GetListOfBranches();
    if (branches == nullptr)
      throw Error("tree \"" + treeName + "\" in file " + fileName + " contains no branches", 7006);

    const size_t n = branches->GetEntries();
    m_branches.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      TBranch* b = dynamic_cast<TBranch*>(branches->At(i));
      if (b == nullptr)
        continue;
      std::string const name(b->GetName());
      if (name == "EventAux")
        continue;
      size_type s = getTotalSize(b);
      size_t compressed_size = s[kCompressed];
      size_t uncompressed_size = s[kUncompressed];
      if (m_mode == Mode::branches) {
        m_branches.push_back(BranchRecord(name, m_nEvents, compressed_size, uncompressed_size));
        std::sort(m_branches.begin(),
                  m_branches.end(),
                  std::bind(std::greater<double>(),
                            std::bind(&BranchRecord::compr_size, std::placeholders::_1),
                            std::bind(&BranchRecord::compr_size, std::placeholders::_2)));
      } else if (m_mode == Mode::leaves) {
        Leaves new_leaves = getLeaves(b);
        m_leaves.insert(m_leaves.end(), new_leaves.begin(), new_leaves.end());

        auto new_leaves_compressed =
            std::accumulate(new_leaves.begin(), new_leaves.end(), 0, [](size_t sum, LeafRecord const& leaf) {
              return sum + leaf.compr_size;
            });
        auto new_leaves_uncompressed =
            std::accumulate(new_leaves.begin(), new_leaves.end(), 0, [](size_t sum, LeafRecord const& leaf) {
              return sum + leaf.uncompr_size;
            });
        size_t overehead_compressed = compressed_size - new_leaves_compressed;
        size_t overehead_uncompressed = uncompressed_size - new_leaves_uncompressed;
        m_leaves.push_back(LeafRecord(name + "overhead", m_nEvents, overehead_compressed, overehead_uncompressed));
        std::sort(m_leaves.begin(),
                  m_leaves.end(),
                  std::bind(std::greater<double>(),
                            std::bind(&LeafRecord::compr_size, std::placeholders::_1),
                            std::bind(&LeafRecord::compr_size, std::placeholders::_2)));
      } else {
        throw Error("Unsupported mode", 7007);
      }
    }
  }

  void EdmEventSize::sortAlpha() {
    if (m_mode == Mode::branches)
      std::sort(m_branches.begin(),
                m_branches.end(),
                std::bind(std::less<std::string>(),
                          std::bind(&BranchRecord::name, std::placeholders::_1),
                          std::bind(&BranchRecord::name, std::placeholders::_2)));
    else if (m_mode == Mode::leaves)
      std::sort(m_leaves.begin(),
                m_leaves.end(),
                std::bind(std::less<std::string>(),
                          std::bind(&LeafRecord::name, std::placeholders::_1),
                          std::bind(&LeafRecord::name, std::placeholders::_2)));
    else {
      throw Error("Unsupported mode", 7007);
    }
  }

  namespace detail {
    // format as product:label (type)
    void shorterName(EdmEventSize::BranchRecord& br) {
      size_t b = br.fullName.find('_');
      size_t e = br.fullName.rfind('_');
      if (b == e)
        br.name = br.fullName;
      else {
        // remove type and process
        br.name = br.fullName.substr(b + 1, e - b - 1);
        // change label separator in :
        e = br.name.rfind('_');
        if (e != std::string::npos)
          br.name.replace(e, 1, ":");
        // add the type name
        br.name.append(" (" + br.fullName.substr(0, b) + ")");
      }
    }

    // format as product:label (type) object (objectType)
    void shorterName(EdmEventSize::LeafRecord& lr) {
      size_t b = lr.branch.find('_');
      size_t e = lr.branch.rfind('_');
      if (b == e)
        lr.name = lr.branch;
      else {
        // remove type and process
        lr.name = lr.branch.substr(b + 1, e - b - 1);
        // change label separator in :
        e = lr.name.rfind('_');
        if (e != std::string::npos)
          lr.name.replace(e, 1, ":");
        // add the type name
        lr.name.append(" (" + lr.branch.substr(0, b) + ")");
      }
      if (!lr.object.empty()) {
        // object is objectName_objectType. Transform in objectName (objectType) and add to name
        e = lr.object.find('|');
        if (e != std::string::npos) {
          std::string obj = lr.object.substr(0, e);
          std::string objType = lr.object.substr(e + 1);
          lr.name.append(" " + obj + " (" + objType + ")");
        } else {
          lr.name.append(" " + lr.object);
        }
      }
    }

  }  // namespace detail

  void EdmEventSize::formatNames() {
    if (m_mode == Mode::branches)
      std::for_each(m_branches.begin(),
                    m_branches.end(),
                    static_cast<void (*)(EdmEventSize::BranchRecord&)>(&detail::shorterName));
    else if (m_mode == Mode::leaves)
      std::for_each(
          m_leaves.begin(), m_leaves.end(), static_cast<void (*)(EdmEventSize::LeafRecord&)>(&detail::shorterName));
    else {
      throw Error("Unsupported mode", 7007);
    }
  }

  namespace detail {

    template <typename Record>
    void dump(std::ostream& co, Record const& record) {
      co << record.name << " " << (double)record.uncompr_size / (double)record.nEvents << " "
         << (double)record.compr_size / (double)record.nEvents << "\n";
    }

    const std::string RESOURCES_JSON = R"(
    "resources": [
    {
    "name": "size_uncompressed",
    "description" : "uncompressed size",
    "unit" : "B",
    "title" : "Data Size"
    },
    {
    "name":"size_compressed",
    "description": "compressed size",
    "unit" : "B",
    "title" : "Data Size"
    }
    ],
    )";

    void dumpJson(std::ostream& co, EdmEventSize::LeafRecord const& lr, bool isLast = false) {
      co << "{\n";
      co << "\"events\": " << lr.nEvents << ",\n";
      co << "\"type\": \"" << lr.branch << "\",\n";
      co << "\"label\": \"" << lr.object << "\",\n";
      co << "\"size_compressed\": " << lr.compr_size << ",\n";
      co << "\"size_uncompressed\": " << lr.uncompr_size << ",\n";
      if (lr.uncompr_size == 0)
        co << "\"ratio\": 0\n";
      else
        co << "\"ratio\": " << (double)lr.compr_size / (double)lr.uncompr_size << "\n";
      if (isLast)
        co << "}\n";
      else
        co << "},\n";
    }

    void dumpJson(std::ostream& co, EdmEventSize::BranchRecord const& br, bool isLast = false) {
      co << "{\n";
      co << "\"events\": " << br.nEvents << ",\n";
      co << "\"type\": \"" << br.name << "\",\n";
      co << "\"size_compressed\": " << br.compr_size << ",\n";
      co << "\"size_uncompressed\": " << br.uncompr_size << ",\n";
      if (br.uncompr_size == 0)
        co << "\"ratio\": 0\n";
      else
        co << "\"ratio\": " << (double)br.compr_size / (double)br.uncompr_size << "\n";
      if (isLast)
        co << "}\n";
      else
        co << "},\n";
    }

  }  // namespace detail

  void EdmEventSize::dump(std::ostream& co, bool header, Format format) const {
    if (format == Format::text) {
      if (header) {
        co << "File " << m_fileName << " Events " << m_nEvents << "\n";
        if (m_mode == Mode::branches) {
          co << "Branch Name | Average Uncompressed Size (Bytes/Event) | Average Compressed Size (Bytes/Event) \n";
        } else if (m_mode == Mode::leaves) {
          co << "Leaf Name | Average Uncompressed Size (Bytes/Event) | Average Compressed Size (Bytes/Event) \n";
        } else {
          throw Error("Unsupported mode", 7007);
        }
      }
      if (m_mode == Mode::branches)
        std::for_each(m_branches.begin(),
                      m_branches.end(),
                      std::bind(detail::dump<BranchRecord>, std::ref(co), std::placeholders::_1));
      else if (m_mode == Mode::leaves)
        std::for_each(
            m_leaves.begin(), m_leaves.end(), std::bind(detail::dump<LeafRecord>, std::ref(co), std::placeholders::_1));
      else {
        throw Error("Unsupported mode", 7007);
      }
    }
    if (format == Format::json) {
      // Modules json
      co << "{\n";
      co << "\"modules\": [\n";
      if (m_mode == Mode::branches) {
        std::for_each(
            m_branches.begin(), m_branches.end() - 1, [&co](const BranchRecord& br) { detail::dumpJson(co, br); });
        detail::dumpJson(co, m_branches.back(), true);
      } else if (m_mode == Mode::leaves) {
        std::for_each(m_leaves.begin(), m_leaves.end() - 1, [&co](const LeafRecord& lr) { detail::dumpJson(co, lr); });
        detail::dumpJson(co, m_leaves.back(), true);
      } else {
        throw Error("Unsupported mode", 7007);
      }
      co << "],\n";

      // Resources json
      co << detail::RESOURCES_JSON;

      // Total json
      co << "\"total\": {\n";
      co << "\"events\": " << m_nEvents << ",\n";
      auto [total_uncompressed, total_compressed] = std::accumulate(
          m_leaves.begin(), m_leaves.end(), std::make_pair<size_t>(0, 0), [](auto sum, LeafRecord const& leaf) {
            return std::make_pair(sum.first + leaf.uncompr_size, sum.second + leaf.compr_size);
          });
      co << "\"size_uncompressed\": " << total_uncompressed << ",\n";
      co << "\"size_compressed\": " << total_compressed << ",\n";
      co << "\"ratio\": "
         << (total_uncompressed == 0 ? 0.0
                                     : static_cast<double>(total_compressed) / static_cast<double>(total_uncompressed))
         << "\n";
      co << "}\n}\n";
    }
  }

  namespace detail {
    struct Hist {
      explicit Hist(int itop)
          : top(itop),
            uncompressed("uncompressed", "sizes", top, -0.5, -0.5 + top),
            compressed("compressed", "sizes", top, -0.5, -0.5 + top),
            cxAxis(compressed.GetXaxis()),
            uxAxis(uncompressed.GetXaxis()),
            x(0) {}

      template <typename Record>
      void fill(Record const& record) {
        if (x < top) {
          cxAxis->SetBinLabel(x + 1, record.name.c_str());
          uxAxis->SetBinLabel(x + 1, record.name.c_str());
          compressed.Fill(x, record.compr_size);
          uncompressed.Fill(x, record.uncompr_size);
          x++;
        }
      }

      void finalize() {
        double mn = std::numeric_limits<double>::max();
        for (int i = 1; i <= top; ++i) {
          double cm = compressed.GetMinimum(i), um = uncompressed.GetMinimum(i);
          if (cm > 0 && cm < mn)
            mn = cm;
          if (um > 0 && um < mn)
            mn = um;
        }
        mn *= 0.8;
        double mx = std::max(compressed.GetMaximum(), uncompressed.GetMaximum());
        mx *= 1.2;
        uncompressed.SetMinimum(mn);
        uncompressed.SetMaximum(mx);
        compressed.SetMinimum(mn);
        //  compressed.SetMaximum( mx );
        cxAxis->SetLabelOffset(-0.32);
        cxAxis->LabelsOption("v");
        cxAxis->SetLabelSize(0.03);
        uxAxis->SetLabelOffset(-0.32);
        uxAxis->LabelsOption("v");
        uxAxis->SetLabelSize(0.03);
        compressed.GetYaxis()->SetTitle("Bytes");
        compressed.SetFillColor(kBlue);
        compressed.SetLineWidth(2);
        uncompressed.GetYaxis()->SetTitle("Bytes");
        uncompressed.SetFillColor(kRed);
        uncompressed.SetLineWidth(2);
      }

      int top;
      TH1F uncompressed;
      TH1F compressed;
      TAxis* cxAxis;
      TAxis* uxAxis;
      int x;
    };

  }  // namespace detail

  void EdmEventSize::produceHistos(std::string const& plot, std::string const& file, int top) const {
    if (top == 0) {
      if (m_mode == Mode::branches)
        top = m_branches.size();
      else if (m_mode == Mode::leaves)
        top = m_leaves.size();
      else
        throw Error("Unsupported mode", 7007);
    }
    detail::Hist h(top);
    if (m_mode == Mode::branches) {
      std::for_each(m_branches.begin(), m_branches.end(), [&h](const BranchRecord& br) { h.fill<BranchRecord>(br); });
    } else if (m_mode == Mode::leaves) {
      h.uncompressed.SetTitle("Leaf sizes");
      h.compressed.SetTitle("Leaf sizes");
      std::for_each(m_leaves.begin(), m_leaves.end(), [&h](const LeafRecord& lr) { h.fill<LeafRecord>(lr); });
    } else
      throw Error("Unsupported mode", 7007);
    h.finalize();
    if (!plot.empty()) {
      gROOT->SetStyle("Plain");
      gStyle->SetOptStat(kFALSE);
      gStyle->SetOptLogy();
      TCanvas c;
      h.uncompressed.Draw();
      h.compressed.Draw("same");
      c.SaveAs(plot.c_str());
    }
    if (!file.empty()) {
      TFile f(file.c_str(), "RECREATE");
      h.compressed.Write();
      h.uncompressed.Write();
      f.Close();
    }
  }

}  // namespace perftools
