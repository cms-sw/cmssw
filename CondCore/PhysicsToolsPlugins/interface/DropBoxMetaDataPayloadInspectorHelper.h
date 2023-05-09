#ifndef DropBoxMetaDataPayloadInspectorHelper_H
#define DropBoxMetaDataPayloadInspectorHelper_H

#include "TH1.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "TLine.h"

#include <fmt/printf.h>
#include <fstream>
#include <boost/tokenizer.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include "CondFormats/Common/interface/DropBoxMetadata.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace DBoxMetadataHelper {
  class RecordMetaDataInfo {
  public:
    /// Constructor
    RecordMetaDataInfo(DropBoxMetadata::Parameters params) {
      const auto& theParameters = params.getParameterMap();
      for (const auto& [key, val] : theParameters) {
        if (key.find("prep") != std::string::npos) {
          m_prepmetadata = val;
        } else if (key.find("prod") != std::string::npos) {
          m_prodmetadata = val;
        } else if (key.find("mult") != std::string::npos) {
          m_multimetadata = val;
        }
      }
    }
    /// Destructor
    ~RecordMetaDataInfo() = default;

  public:
    const std::string getPrepMetaData() const { return m_prepmetadata; }
    const std::string getProdMetaData() const { return m_prodmetadata; }
    const std::string getMultiMetaData() const { return m_multimetadata; }
    const bool hasMultiMetaData() const { return !m_multimetadata.empty(); }

  private:
    std::string m_prepmetadata;
    std::string m_prodmetadata;
    std::string m_multimetadata;
  };

  using recordMap = std::map<std::string, RecordMetaDataInfo>;

  inline const std::vector<std::string> getAllRecords(const DBoxMetadataHelper::recordMap& recordSet) {
    std::vector<std::string> records;
    std::transform(recordSet.begin(),
                   recordSet.end(),
                   std::inserter(records, records.end()),
                   [](std::pair<std::string, DBoxMetadataHelper::RecordMetaDataInfo> recordSetEntry) -> std::string {
                     return recordSetEntry.first;
                   });
    return records;
  }

  inline std::vector<std::string> set_difference(std::vector<std::string> const& v1,
                                                 std::vector<std::string> const& v2) {
    std::vector<std::string> diff;
    std::set_difference(std::begin(v1), std::end(v1), std::begin(v2), std::end(v2), std::back_inserter(diff));
    return diff;
  }

  inline std::vector<std::string> set_intersection(std::vector<std::string> const& v1,
                                                   std::vector<std::string> const& v2) {
    std::vector<std::string> common;
    std::set_intersection(std::begin(v1), std::end(v1), std::begin(v2), std::end(v2), std::back_inserter(common));
    return common;
  }

  class DBMetaDataTableDisplay {
  public:
    DBMetaDataTableDisplay(DBoxMetadataHelper::recordMap theMap) : m_Map(theMap) {}
    ~DBMetaDataTableDisplay() = default;

    void printMetaDatas() {
      for (const auto& [key, val] : m_Map) {
        edm::LogPrint("DropBoxMetadataPIHelper") << "key: " << key;
        edm::LogPrint("DropBoxMetadataPIHelper") << "prep: " << cleanJson(val.getPrepMetaData());
        edm::LogPrint("DropBoxMetadataPIHelper") << "prod: " << cleanJson(val.getProdMetaData());
        // check, since it's optional
        if (val.hasMultiMetaData()) {
          edm::LogPrint("DropBoxMetadataPIHelper") << "multi: " << cleanJson(val.getMultiMetaData());
        }
        edm::LogPrint("DropBoxMetadataPIHelper") << "\n";
      }
    }

    void printOneKey(const DBoxMetadataHelper::RecordMetaDataInfo& oneKey) {
      edm::LogPrint("DropBoxMetadataPIHelper") << "prep: " << cleanJson(oneKey.getPrepMetaData());
      edm::LogPrint("DropBoxMetadataPIHelper") << "prod: " << cleanJson(oneKey.getProdMetaData());
      // check, since it's optional
      if (oneKey.hasMultiMetaData()) {
        edm::LogPrint("DropBoxMetadataPIHelper") << "multi: " << cleanJson(oneKey.getMultiMetaData());
      }
      edm::LogPrint("DropBoxMetadataPIHelper") << "\n";
    }

    void printDiffWithMetadata(const DBoxMetadataHelper::recordMap& theRefMap) {
      edm::LogPrint("DropBoxMetadataPIHelper")
          << "Target has: " << m_Map.size() << " records, reference has: " << theRefMap.size() << " records";

      const auto& ref_records = DBoxMetadataHelper::getAllRecords(theRefMap);
      const auto& tar_records = DBoxMetadataHelper::getAllRecords(m_Map);

      const auto& diff = DBoxMetadataHelper::set_difference(ref_records, tar_records);
      const auto& common = DBoxMetadataHelper::set_intersection(ref_records, tar_records);

      // do first the common parts
      for (const auto& key : common) {
        edm::LogPrint("DropBoxMetadataPIHelper") << "key: " << key;
        const auto& val = m_Map.at(key);
        const auto& refval = theRefMap.at(key);

        if ((val.getPrepMetaData()).compare(refval.getPrepMetaData()) != 0) {
          edm::LogPrint("DropBoxMetadataPIHelper") << "found difference in prep metadata!";
          edm::LogPrint("DropBoxMetadataPIHelper") << " in target   : " << cleanJson(val.getPrepMetaData());
          edm::LogPrint("DropBoxMetadataPIHelper") << " in reference: " << cleanJson(refval.getPrepMetaData());
        }
        if ((val.getProdMetaData()).compare(refval.getProdMetaData()) != 0) {
          edm::LogPrint("DropBoxMetadataPIHelper") << "found difference in prod metadata!";
          edm::LogPrint("DropBoxMetadataPIHelper") << " in target   : " << cleanJson(val.getProdMetaData());
          edm::LogPrint("DropBoxMetadataPIHelper") << " in reference: " << cleanJson(refval.getProdMetaData());
        }
        if ((val.getMultiMetaData()).compare(refval.getMultiMetaData()) != 0) {
          edm::LogPrint("DropBoxMetadataPIHelper") << "found difference in multi metadata!";
          edm::LogPrint("DropBoxMetadataPIHelper") << " in target   : " << cleanJson(val.getMultiMetaData());
          edm::LogPrint("DropBoxMetadataPIHelper") << " in reference: " << cleanJson(refval.getMultiMetaData());
        }
        edm::LogPrint("DropBoxMetadataPIHelper") << "\n";
      }

      // if interesction is not the union check for extra differences
      if (!diff.empty()) {
        // check if the reference has more records than target
        if (ref_records.size() > tar_records.size()) {
          for (const auto& ref : ref_records) {
            if (std::find(tar_records.begin(), tar_records.end(), ref) == tar_records.end()) {
              const auto& refval = theRefMap.at(ref);
              edm::LogPrint("DropBoxMetadataPIHelper") << "key: " << ref << " not present in target!";
              printOneKey(refval);
            }
          }
        }
        // then check if the target has more records than the reference
        else if (tar_records.size() > ref_records.size()) {
          for (const auto& tar : tar_records) {
            if (std::find(ref_records.begin(), ref_records.end(), tar) == ref_records.end()) {
              const auto& tarval = m_Map.at(tar);
              edm::LogPrint("DropBoxMetadataPIHelper") << "key: " << tar << " not present in reference!";
              printOneKey(tarval);
            }
          }
        }
      }
    }

  private:
    DBoxMetadataHelper::recordMap m_Map;

    std::string replaceAll(std::string str, const std::string& from, const std::string& to) {
      size_t start_pos = 0;
      while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();  // Handles case where 'to' is a substring of 'from'
      }
      return str;
    }

    std::string cleanJson(std::string str) {
      std::string out = replaceAll(str, std::string("&quot;"), std::string("'"));
      return out;
    }
  };

  class DBMetaDataPlotDisplay {
  public:
    DBMetaDataPlotDisplay(DBoxMetadataHelper::recordMap theMap, std::string theTag, std::string theIOV)
        : m_Map(theMap), m_tagName(theTag), m_IOVsinceDisplay(theIOV) {}
    ~DBMetaDataPlotDisplay() = default;

    void setImageFileName(const std::string& theFileName) {
      m_imageFileName = theFileName;
      return;
    }

    //___________________________________________________________________
    void plotMetaDatas() {
      unsigned int mapsize = m_Map.size();
      float pitch = 1. / (mapsize * 1.1);

      float y, x1, x2;
      std::vector<float> y_x1, y_x2, y_line;
      std::vector<std::string> s_x1, s_x2, s_x3;

      // starting table at y=1.0 (top of the canvas)
      // first column is at 0.02, second column at 0.32 NDC
      y = 1.0;
      x1 = 0.02;
      x2 = x1 + 0.30;

      y -= pitch;
      y_x1.push_back(y);
      s_x1.push_back("#scale[1.2]{Key}");
      y_x2.push_back(y);
      s_x2.push_back("#scale[1.2]{tag: " + m_tagName + " in IOV: " + m_IOVsinceDisplay + "}");

      y -= pitch / 2.;
      y_line.push_back(y);

      for (const auto& element : m_Map) {
        y -= pitch;
        y_x1.push_back(y);
        s_x1.push_back(element.first);

        std::vector<std::string> output;
        std::string toAppend = "";

        std::string prepMetaData = element.second.getPrepMetaData();
        std::string prodMetaData = element.second.getProdMetaData();

        // Remove &quot and uninteresting text from output for sake of clarity
        cleanPrepString(prepMetaData);
        cleanProdString(prodMetaData);

        const std::vector<std::string> pathsPrep = decompose(prepMetaData);
        const std::vector<std::string> pathsProd = decompose(prodMetaData);

        const int colWidth = 80;

        toAppend = "PREP: ";
        output.push_back(toAppend);
        toAppend.clear();
        for (unsigned int iPath = 0; iPath < pathsPrep.size(); ++iPath) {
          const std::string& thisString = pathsPrep[iPath];
          // skip userText since we want to see actual contents, not metadata
          if (thisString.find("userText") == std::string::npos) {
            // if the line to be added has less than colWidth chars, and is not a new tag ("inputTag"), append to current
            if ((toAppend + thisString).length() < colWidth && thisString.find("inputTag") != 0) {
              toAppend += thisString;
            } else {
              // else if the line exceeds colWidth chars or this is a new tag ("inputTag"), dump in the vector and resume from scratch
              output.push_back(toAppend);
              toAppend.clear();
              toAppend += thisString;
            }
          }
          // if it's the last, dump it
          if (iPath == pathsPrep.size() - 1) {
            output.push_back(toAppend);
          }
        }

        toAppend = "PROD: ";
        output.push_back(toAppend);
        toAppend.clear();
        for (unsigned int iPath = 0; iPath < pathsProd.size(); ++iPath) {
          const std::string& thisString = pathsProd[iPath];

          if (thisString.find("userText") == std::string::npos) {
            // if the line to be added has less than colWidth chars append to current
            if ((toAppend + thisString).length() < colWidth && thisString.find("inputTag") != 0) {
              toAppend += thisString;
            } else {
              // else if the line exceeds colWidth chars or this is a new tag ("inputTag"), dump in the vector and resume from scratch
              output.push_back(toAppend);
              toAppend.clear();
              toAppend += thisString;
            }
          }
          // if it's the last, dump it
          if (iPath == pathsProd.size() - 1)
            output.push_back(toAppend);
        }

        for (unsigned int br = 0; br < output.size(); br++) {
          y_x2.push_back(y);
          // do not use red color since colors get mixed if output[br]
          // contains a right curly brace (I could not find a way to circumvent that)
          s_x2.push_back(output[br]);

          if (br != output.size() - 1)
            y -= pitch;
        }

        y_line.push_back(y - (pitch / 2.));
      }

      TCanvas canvas("DropBoxMetaData", "DropBoxMetaData", 2000, std::max(y_x1.size(), y_x2.size()) * 40);
      TLatex l;

      // Draw the columns titles
      l.SetTextAlign(12);
      float newpitch = 1 / (std::max(y_x1.size(), y_x2.size()) * 1.1);
      float factor = newpitch / pitch;
      l.SetTextSize(newpitch - 0.002);
      canvas.cd();
      for (unsigned int i = 0; i < y_x1.size(); i++) {
        l.DrawLatexNDC(x1, 1 - (1 - y_x1[i]) * factor, s_x1[i].c_str());
      }

      for (unsigned int i = 0; i < y_x2.size(); i++) {
        l.DrawLatexNDC(x2, 1 - (1 - y_x2[i]) * factor, s_x2[i].c_str());
      }
      canvas.cd();
      canvas.Update();

      // Draw horizontal lines separating records
      TLine lines[y_line.size()];
      unsigned int iL = 0;
      for (const auto& line : y_line) {
        lines[iL] = TLine(gPad->GetUxmin(), 1 - (1 - line) * factor, gPad->GetUxmax(), 1 - (1 - line) * factor);
        lines[iL].SetLineWidth(1);
        lines[iL].SetLineStyle(9);
        lines[iL].SetLineColor(2);
        lines[iL].Draw("same");
        iL++;
      }

      std::string fileName("DropBoxMetadata_Display.png");
      if (!m_imageFileName.empty())
        fileName = m_imageFileName;
      canvas.SaveAs(fileName.c_str());
    }

    //___________________________________________________________________
    void plotDiffWithMetadata(const DBoxMetadataHelper::recordMap& theRefMap,
                              std::string theRefTag,
                              std::string theRefIOV) {
      const auto& ref_records = DBoxMetadataHelper::getAllRecords(theRefMap);
      const auto& tar_records = DBoxMetadataHelper::getAllRecords(m_Map);

      //      const auto& diff = DBoxMetadataHelper::set_difference(ref_records, tar_records);
      const auto& common = DBoxMetadataHelper::set_intersection(ref_records, tar_records);

      // preparations for plotting
      // starting table at y=1.0 (top of the canvas)
      // first column is at 0.03, second column at 0.22 NDC
      unsigned int mapsize = 2 * m_Map.size();
      float pitch = 1. / (mapsize * 1.1);
      float y, x1, x2;
      std::vector<float> y_x1, y_x2, y_line;
      std::vector<std::string> s_x1, s_x2, s_x3;
      y = 1.0;
      x1 = 0.02;
      x2 = x1 + 0.30;
      y -= pitch;

      // title for plot
      y_x1.push_back(y);
      s_x1.push_back("#scale[1.2]{Key}");
      y_x2.push_back(y);
      s_x2.push_back("#scale[1.2]{Target tag / IOV :" + m_tagName + " / " + m_IOVsinceDisplay + "}");

      y -= pitch;
      y_x1.push_back(y);
      s_x1.push_back("");
      y_x2.push_back(y);
      s_x2.push_back("#scale[1.2]{Reference tag / IOV :" + theRefTag + " / " + theRefIOV + "}");

      y -= pitch / 2.;
      y_line.push_back(y);

      // do first the common parts
      for (const auto& key : common) {
        const auto& val = m_Map.at(key);
        const auto& refval = theRefMap.at(key);

        y -= pitch;
        y_x1.push_back(y);
        s_x1.push_back(key);

        std::vector<std::string> output;

        std::string tarPrepMetaData = val.getPrepMetaData();
        std::string tarProdMetaData = val.getProdMetaData();
        std::string refPrepMetaData = refval.getPrepMetaData();
        std::string refProdMetaData = refval.getProdMetaData();

        // Remove &quot and uninteresting text from output for sake of clarity
        cleanPrepString(tarPrepMetaData);
        cleanPrepString(refPrepMetaData);
        cleanProdString(tarProdMetaData);
        cleanProdString(refProdMetaData);

        const std::vector<std::string> tarPathsPrep = decompose(tarPrepMetaData);
        const std::vector<std::string> refPathsPrep = decompose(refPrepMetaData);
        const std::vector<std::string> tarPathsProd = decompose(tarProdMetaData);
        const std::vector<std::string> refPathsProd = decompose(refProdMetaData);

        bool refAndTarIdentical = true;
        std::string tmpTar = "";
        std::string tmpRef = "";

        prepareLine(tarPathsPrep, output, tmpTar, "PREP/tar");
        prepareLine(refPathsPrep, output, tmpRef, "PREP/ref");

        // check if printouts are identical for PREP
        eraseAllSubStr(tmpTar, "PREP/tar: ");
        eraseAllSubStr(tmpRef, "PREP/ref: ");
        if (tmpTar != tmpRef) {
          refAndTarIdentical = false;
        } else {
          output.clear();
        }

        // determine the size after having filled the prep- metadata
        size_t lenAfterPrep = output.size();

        // clear the tmps
        tmpTar = "";
        tmpRef = "";

        prepareLine(tarPathsProd, output, tmpTar, "PROD/tar");
        prepareLine(refPathsProd, output, tmpRef, "PROD/ref");

        // check if printouts are identical for PROD
        eraseAllSubStr(tmpTar, "PROD/tar: ");
        eraseAllSubStr(tmpRef, "PROD/ref: ");
        if (tmpTar != tmpRef) {
          refAndTarIdentical = false;
        } else {
          // remove everything after the prep one
          output.erase(output.end() - lenAfterPrep, output.end());
        }

        // print either "identical" or contents of tags
        if (refAndTarIdentical) {
          y_x2.push_back(y);
          s_x2.push_back("#color[4]{identical}");
        } else {
          for (unsigned int br = 0; br < output.size(); br++) {
            y_x2.push_back(y);
            // do not use red color since colors get mixed if output[br]
            // contains a right curly brace (I could not find a way to circumvent that)
            s_x2.push_back(output[br]);

            if (br != output.size() - 1)
              y -= pitch;
          }
        }
        y_line.push_back(y - (pitch / 2.));
      }

      // now when common parts are handled, check if there are additional records
      // (one could check if diff is empty, but his doesn't seem to work as expected)

      // First, check if there are records in reference which are not in target
      for (const auto& ref : ref_records) {
        if (std::find(tar_records.begin(), tar_records.end(), ref) == tar_records.end()) {
          y -= pitch;
          y_x1.push_back(y);
          s_x1.push_back(ref);
          y_x2.push_back(y);
          s_x2.push_back("#bf{Only in reference, not in target.}");
          y_line.push_back(y - (pitch / 2.));
        }
      }

      // Second, check if there are records in target which are not in reference
      for (const auto& tar : tar_records) {
        if (std::find(ref_records.begin(), ref_records.end(), tar) == ref_records.end()) {
          y -= pitch;
          y_x1.push_back(y);
          s_x1.push_back(tar);
          y_x2.push_back(y);
          s_x2.push_back("#bf{Only in target, not in reference.}");
          y_line.push_back(y - (pitch / 2.));
        }
      }

      // Finally, print text to TCanvas

      TCanvas canvas("DropBoxMetaData", "DropBoxMetaData", 2000, std::max(y_x1.size(), y_x2.size()) * 40);
      TLatex l;
      // Draw the columns titles
      l.SetTextAlign(12);

      float newpitch = 1 / (std::max(y_x1.size(), y_x2.size()) * 1.1);
      float factor = newpitch / pitch;
      l.SetTextSize(newpitch - 0.002);
      canvas.cd();
      for (unsigned int i = 0; i < y_x1.size(); i++) {
        l.DrawLatexNDC(x1, 1 - (1 - y_x1[i]) * factor, s_x1[i].c_str());
      }

      for (unsigned int i = 0; i < y_x2.size(); i++) {
        l.DrawLatexNDC(x2, 1 - (1 - y_x2[i]) * factor, s_x2[i].c_str());
      }

      canvas.cd();
      canvas.Update();

      // Draw horizontal lines separating records
      TLine lines[y_line.size()];
      unsigned int iL = 0;
      for (const auto& line : y_line) {
        lines[iL] = TLine(gPad->GetUxmin(), 1 - (1 - line) * factor, gPad->GetUxmax(), 1 - (1 - line) * factor);
        lines[iL].SetLineWidth(1);
        lines[iL].SetLineStyle(9);
        lines[iL].SetLineColor(2);
        lines[iL].Draw("same");
        iL++;
      }

      std::string fileName("DropBoxMetadata_Compare.png");
      if (!m_imageFileName.empty())
        fileName = m_imageFileName;
      canvas.SaveAs(fileName.c_str());
    }

  private:
    DBoxMetadataHelper::recordMap m_Map;  //!< map of the record / metadata associations
    std::string m_tagName;                //!< tag name
    std::string m_IOVsinceDisplay;        //!< iov since
    std::string m_imageFileName;          //!< image file name

    //___________________________________________________________________
    std::string replaceAll(std::string str, const std::string& from, const std::string& to) {
      size_t start_pos = 0;
      while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();  // Handles case where 'to' is a substring of 'from'
      }
      return str;
    }

    //___________________________________________________________________
    std::string cleanJson(std::string str) {
      std::string out = replaceAll(str, std::string("&quot;"), std::string("'"));
      return out;
    }

    //___________________________________________________________________
    void eraseAllSubStr(std::string& s, const std::string& toErase) {
      size_t pos = std::string::npos;
      // Search for the substring in string in a loop until nothing is found
      while ((pos = s.find(toErase)) != std::string::npos) {
        // If found then erase it from string
        s.erase(pos, toErase.length());
      }
      return;
    }

    //___________________________________________________________________
    void cleanPrepString(std::string& myString) {
      eraseAllSubStr(myString, "&quot;");
      eraseAllSubStr(myString, "destinationDatabase: oracle://cms_orcoff_prep/CMS_CONDITIONS, ");
      eraseAllSubStr(myString, "since: null, ");
      eraseAllSubStr(myString, "{");
      eraseAllSubStr(myString, "}");
      eraseAllSubStr(myString, ":");
      myString = replaceAll(myString, "destinationTags", "destinationTags:");
      myString = replaceAll(myString, "inputTag", "inputTag:");
      return;
    }

    //___________________________________________________________________
    void cleanProdString(std::string& myString) {
      eraseAllSubStr(myString, "&quot;");
      eraseAllSubStr(myString, "destinationDatabase: oracle://cms_orcon_prod/CMS_CONDITIONS, ");
      eraseAllSubStr(myString, "since: null, ");
      eraseAllSubStr(myString, "{");
      eraseAllSubStr(myString, "}");
      eraseAllSubStr(myString, ":");
      myString = replaceAll(myString, "destinationTags", "destinationTags:");
      myString = replaceAll(myString, "inputTag", "inputTag:");
      return;
    }

    //___________________________________________________________________
    std::vector<std::string> decompose(const std::string& s) const {
      // decompose 's' into its parts that are separated by 'delimeter_'
      // (similar as in
      //  Alignment/CommonAlignmentAlgorithm/src/AlignmentParameterSelector.cc)

      const std::string::value_type delimeter_ = ',';  // separator
      const std::string::value_type space_ = ' ';      // separator

      std::vector<std::string> result;
      if (!(s.size() == 1 && s[0] == delimeter_)) {
        // delimeter_ only indicates an empty list as DB cannot store empty strings
        std::string::size_type previousPos = 0;
        while (true) {
          const std::string::size_type delimiterPos = s.find(delimeter_, previousPos);
          if (delimiterPos == std::string::npos) {
            result.push_back(s.substr(previousPos));  // until end
            break;
          }
          result.push_back(s.substr(previousPos, delimiterPos - previousPos));
          previousPos = delimiterPos + 1;  // +1: skip delim
          if (s[previousPos] == space_)
            previousPos++;  // remove space
        }
      }
      return result;
    }

    //___________________________________________________________________
    void prepareLine(const std::vector<std::string>& thePaths,
                     std::vector<std::string>& output,
                     std::string& tmp,
                     const std::string& header) {
      const int color = (header.find("tar") == std::string::npos) ? 2 /*kRed*/ : 3 /*kGreen*/;
      const int colWidth = 80;  // maximum width of column

      std::string toAppend = "";
      toAppend = header;
      output.push_back("#color[" + std::to_string(color) + "]{" + toAppend + "}");
      toAppend.clear();
      for (unsigned int iPath = 0; iPath < thePaths.size(); ++iPath) {
        const std::string& thisString = thePaths[iPath];
        // skip userText since we want to compare actual contents, not metadata
        if (thisString.find("userText") == std::string::npos) {
          // if the line to be added has less than colWidth chars, and is not a new tag ("inputTag"), append to current
          if ((toAppend + thisString).length() < colWidth && thisString.find("inputTag") != 0) {
            toAppend += thisString;
          } else {
            // else if the line exceeds colWidth chars or this is a new tag ("inputTag"), dump in the vector and resume from scratch
            output.push_back("#color[" + std::to_string(color) + "]{" + toAppend + "}");
            tmp += toAppend;
            toAppend.clear();
            toAppend += thisString;
          }
        }
        if (iPath == thePaths.size() - 1) {
          output.push_back("#color[" + std::to_string(color) + "]{" + toAppend + "}");
          tmp += toAppend;
        }
      }
    }
  };
}  // namespace DBoxMetadataHelper
#endif
