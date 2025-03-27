/** \file PerfTools/EdmEvent/interface/EdmEventSize.cc
 *
 *  \author Vincenzo Innocente
 *  \author Simone Rossi Tisbeni
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

namespace perftools {

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

  template <EdmEventMode M>
  using Record = EdmEventSize<M>::Record;

  template <EdmEventMode M>
  EdmEventSize<M>::EdmEventSize() : m_nEvents(0) {}

  template <EdmEventMode M>
  EdmEventSize<M>::EdmEventSize(std::string const& fileName, std::string const& treeName) : m_nEvents(0) {
    parseFile(fileName, treeName);
  }

  template <EdmEventMode M>
  typename EdmEventSize<M>::Records getLeaves(TBranch* b) {
    typename EdmEventSize<M>::Records new_leaves;
    auto subBranches = b->GetListOfBranches();
    const size_t nl = subBranches->GetEntries();
    if (nl == 0) {
      TLeaf* l = dynamic_cast<TLeaf*>(b->GetListOfLeaves()->At(0));
      if (l == nullptr)
        return new_leaves;

      std::string const leaf_name = l->GetName();
      std::string const leaf_type = l->GetTypeName();
      size_t compressed_size = l->GetBranch()->GetZipBytes();
      size_t uncompressed_size = l->GetBranch()->GetTotBytes();
      std::string full_name = leaf_name + '|' + leaf_type;
      full_name.erase(std::remove(full_name.begin(), full_name.end(), ' '), full_name.end());
      size_t nEvents = l->GetBranch()->GetEntries();
      new_leaves.push_back(Record<M>(full_name, nEvents, compressed_size, uncompressed_size));
    } else {
      for (size_t j = 0; j < nl; ++j) {
        TBranch* subBranch = dynamic_cast<TBranch*>(subBranches->At(j));
        if (subBranch == nullptr)
          continue;
        auto leaves = getLeaves<M>(subBranch);
        new_leaves.insert(new_leaves.end(), leaves.begin(), leaves.end());
      }
    }
    return new_leaves;
  }

  template <EdmEventMode M>
  void EdmEventSize<M>::parseFile(std::string const& fileName, std::string const& treeName) {
    m_fileName = fileName;
    m_records.clear();

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
    m_records.reserve(n);
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
      if constexpr (M == EdmEventMode::Branches) {
        m_records.push_back(Record(name, m_nEvents, compressed_size, uncompressed_size));
      } else if constexpr (M == EdmEventMode::Leaves) {
        Records new_leaves = getLeaves<M>(b);
        m_records.insert(m_records.end(), new_leaves.begin(), new_leaves.end());

        auto new_leaves_compressed =
            std::accumulate(new_leaves.begin(), new_leaves.end(), 0, [](size_t sum, Record const& leaf) {
              return sum + leaf.compr_size;
            });
        auto new_leaves_uncompressed =
            std::accumulate(new_leaves.begin(), new_leaves.end(), 0, [](size_t sum, Record const& leaf) {
              return sum + leaf.uncompr_size;
            });
        size_t overehead_compressed = compressed_size - new_leaves_compressed;
        size_t overehead_uncompressed = uncompressed_size - new_leaves_uncompressed;
        m_records.push_back(Record(name + "overhead", m_nEvents, overehead_compressed, overehead_uncompressed));
      } else {
        throw Error("Unsupported mode", 7007);
      }
    }
    std::sort(m_records.begin(),
              m_records.end(),
              std::bind(std::greater<size_t>(),
                        std::bind(&Record::compr_size, std::placeholders::_1),
                        std::bind(&Record::compr_size, std::placeholders::_2)));
  }

  template <EdmEventMode M>
  void EdmEventSize<M>::sortAlpha() {
    std::sort(m_records.begin(),
              m_records.end(),
              std::bind(std::less<std::string>(),
                        std::bind(&Record::name, std::placeholders::_1),
                        std::bind(&Record::name, std::placeholders::_2)));
  }

  namespace detail {
    // format as product:label (type)
    template <EdmEventMode M>
    void shorterName(Record<M>& record) {
      if constexpr (M == EdmEventMode::Branches) {
        std::string const& fullName = record.name;
        size_t b = fullName.find('_');
        size_t e = fullName.rfind('_');
        if (b == e)
          record.name = fullName;
        else {
          // remove type and process
          record.name = fullName.substr(b + 1, e - b - 1);
          // change label separator in :
          e = record.name.rfind('_');
          if (e != std::string::npos)
            record.name.replace(e, 1, ":");
          // add the type name
          record.name.append(" (" + fullName.substr(0, b) + ")");
        }
      } else if constexpr (M == EdmEventMode::Leaves) {
        size_t b = record.type.find('_');
        size_t e = record.type.rfind('_');
        if (b == e)
          record.name = record.type;
        else {
          // remove type and process
          record.name = record.type.substr(b + 1, e - b - 1);
          // change label separator in :
          e = record.name.rfind('_');
          if (e != std::string::npos)
            record.name.replace(e, 1, ":");
          // add the type name
          record.name.append(" (" + record.type.substr(0, b) + ")");
        }
        if (!record.label.empty()) {
          // object is objectName_objectType. Transform in objectName (objectType) and add to name
          e = record.label.find('|');
          if (e != std::string::npos) {
            std::string obj = record.label.substr(0, e);
            std::string objType = record.label.substr(e + 1);
            record.name.append(" " + obj + " (" + objType + ")");
          } else {
            record.name.append(" " + record.label);
          }
        }
      } else {
        throw EdmEventSize<M>::Error("Unsupported mode", 7007);
      }
    }

  }  // namespace detail

  template <EdmEventMode M>
  void EdmEventSize<M>::formatNames() {
    std::for_each(m_records.begin(), m_records.end(), std::bind(detail::shorterName<M>, std::placeholders::_1));
  }

  namespace detail {

    template <EdmEventMode M>
    void dump(std::ostream& co, Record<M> const& record) {
      co << record.name << " " << static_cast<double>(record.uncompr_size) / static_cast<double>(record.nEvents) << " "
         << static_cast<double>(record.compr_size) / static_cast<double>(record.nEvents) << "\n";
    }

    const std::string RESOURCES_JSON = R"("resources": [
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

    template <EdmEventMode M>
    void dumpJson(std::ostream& co, Record<M> const& record, bool isLast = false) {
      co << "{\n";
      co << "\"events\": " << record.nEvents << ",\n";
      co << "\"type\": \"" << record.type << "\",\n";
      co << "\"label\": \"" << record.label << "\",\n";
      co << "\"size_compressed\": " << record.compr_size << ",\n";
      co << "\"size_uncompressed\": " << record.uncompr_size << ",\n";
      co << "\"ratio\": "
         << (record.uncompr_size == 0
                 ? 0.0
                 : static_cast<double>(record.compr_size) / static_cast<double>(record.uncompr_size));
      co << (isLast ? "}\n" : "},\n");
    }

  }  // namespace detail

  template <EdmEventMode M>
  void EdmEventSize<M>::dump(std::ostream& co, bool header) const {
    if (header) {
      co << "File " << m_fileName << " Events " << m_nEvents << "\n";
      if constexpr (M == EdmEventMode::Branches) {
        co << "Branch Name | Average Uncompressed Size (Bytes/Event) | Average Compressed Size (Bytes/Event) \n";
      } else if constexpr (M == EdmEventMode::Leaves) {
        co << "Leaf Name | Average Uncompressed Size (Bytes/Event) | Average Compressed Size (Bytes/Event) \n";
      } else {
        throw Error("Unsupported mode", 7007);
      }
    }

    std::for_each(m_records.begin(), m_records.end(), std::bind(detail::dump<M>, std::ref(co), std::placeholders::_1));
  }

  template <EdmEventMode M>
  void EdmEventSize<M>::dumpJson(std::ostream& co) const {
    // Modules json
    co << "{\n";
    co << "\"modules\": [\n";

    std::for_each(
        m_records.begin(), m_records.end() - 1, [&co](const Record& record) { detail::dumpJson<M>(co, record); });
    detail::dumpJson<M>(co, m_records.back(), true);

    co << "],\n";

    // Resources json
    co << detail::RESOURCES_JSON;

    // Total json
    co << "\"total\": {\n";
    co << "\"events\": " << m_nEvents << ",\n";
    auto [total_uncompressed, total_compressed] = std::accumulate(
        m_records.begin(), m_records.end(), std::make_pair<size_t, size_t>(0, 0), [](auto sum, Record const& leaf) {
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

  namespace detail {
    struct Hist {
      explicit Hist(int itop)
          : top(itop),
            uncompressed("uncompressed", "sizes", top, -0.5, -0.5 + top),
            compressed("compressed", "sizes", top, -0.5, -0.5 + top),
            cxAxis(compressed.GetXaxis()),
            uxAxis(uncompressed.GetXaxis()),
            x(0) {}

      template <EdmEventMode M>
      void fill(Record<M> const& record) {
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

  template <EdmEventMode M>
  void EdmEventSize<M>::produceHistos(std::string const& plot, std::string const& file, int top) const {
    if (top == 0)
      top = m_records.size();

    detail::Hist h(top);
    if constexpr (M == EdmEventMode::Leaves) {
      h.uncompressed.SetTitle("Leaf sizes");
      h.compressed.SetTitle("Leaf sizes");
    }
    std::for_each(
        m_records.begin(), m_records.end(), std::bind(&detail::Hist::fill<M>, std::ref(h), std::placeholders::_1));

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

  template class perftools::EdmEventSize<perftools::EdmEventMode::Leaves>;
  template class perftools::EdmEventSize<perftools::EdmEventMode::Branches>;
}  // namespace perftools
