#ifndef _geometry_hgcalmapping_hgcalmappingtools_h_
#define _geometry_hgcalmapping_hgcalmappingtools_h_

#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include <map>
#include <algorithm>

namespace hgcal {

  namespace mappingtools {

    /**
     * @class HGCalEntityList is a helper class is used to hold the values of the different rows
     * as strings. The class builds its structure and indexing from parsing ascii files separated by spaces
     * assuming the first row is the header
    */
    class HGCalEntityList {
    public:
      typedef std::string HGCalEntityAttr;
      typedef std::vector<HGCalEntityAttr> HGCalEntityRow;

      HGCalEntityList() {}

      /**
           * @short builds the entity list from a file
          */
      void buildFrom(std::string url);

      /**
           * @short gets the attribute corresponding the column col in a row
          */
      HGCalEntityAttr getAttr(std::string col, HGCalEntityRow &row) {
        auto it = columnIndex_.find(col);
        if (it == columnIndex_.end()) {
          throw cms::Exception("ValueError") << "Request for unknown column " << col;
        }
        return row[it->second];
      }
      float getFloatAttr(std::string col, HGCalEntityRow &row) { return (float)atof(getAttr(col, row).c_str()); }
      float getIntAttr(std::string col, HGCalEntityRow &row) { return atoi(getAttr(col, row).c_str()); }
      const std::vector<HGCalEntityRow> &getEntries() { return entities_; }
      HGCalEntityRow getColumnNames() { return colNames_; }
      bool hasColumn(std::string col) { return std::find(colNames_.begin(), colNames_.end(), col) != colNames_.end(); }

    private:
      HGCalEntityRow colNames_;
      std::map<HGCalEntityAttr, size_t> columnIndex_;
      std::vector<HGCalEntityRow> entities_;
      void setHeader(HGCalEntityRow &header) {
        for (size_t i = 0; i < header.size(); i++) {
          std::string cname = header[i];
          colNames_.push_back(cname);
          columnIndex_[cname] = i;
        }
      }
      void addRow(HGCalEntityRow &row) { entities_.push_back(row); }
    };

    uint16_t getEcondErx(uint16_t chip, uint16_t half);
    uint32_t getElectronicsId(
        bool zside, uint16_t fedid, uint16_t captureblock, uint16_t econdidx, int cellchip, int cellhalf, int cellseq);
    uint32_t getSiDetId(bool zside, int moduleplane, int moduleu, int modulev, int celltype, int celliu, int celliv);
    uint32_t getSiPMDetId(bool zside, int moduleplane, int modulev, int celltype, int celliu, int celliv);

    /**
     * @short matches the module and cell info by detId and returns their indices (-1 is used in case index was not found)
     */
    template <class T1, class T2>
    std::pair<int32_t, int32_t> getModuleCellIndicesForSiCell(const T1 &modules, const T2 &cells, uint32_t detid) {
      std::pair<int32_t, int32_t> key(-1, -1);

      //get the module and cell parts of the id
      HGCSiliconDetId siid(detid);

      // match module det id
      uint32_t modid = siid.moduleId().rawId();
      for (int i = 0; i < modules.view().metadata().size(); i++) {
        auto imod = modules.view()[i];
        if (imod.detid() != modid)
          continue;

        key.first = i;

        //match cell by type of module and by cell det id
        DetId::Detector det(DetId::Detector::HGCalEE);
        uint32_t cellid = 0x3ff & HGCSiliconDetId(det, 0, 0, 0, 0, 0, siid.cellU(), siid.cellV()).rawId();
        for (int j = 0; j < cells.view().metadata().size(); j++) {
          auto jcell = cells.view()[j];

          if (jcell.typeidx() != imod.typeidx())
            continue;
          if (jcell.detid() != cellid)
            continue;

          key.second = j;
          return key;
        }

        return key;
      }

      return key;
    }

    /**
     * @short after getting the module/cell indices it returns the sum of the corresponding electronics ids
    */
    template <class T1, class T2>
    uint32_t getElectronicsIdForSiCell(const T1 &modules, const T2 &cells, uint32_t detid) {
      std::pair<int32_t, int32_t> idx = getModuleCellIndicesForSiCell<T1, T2>(modules, cells, detid);
      if (idx.first < 0 || idx.first < 0)
        return 0;
      return modules.view()[idx.first].eleid() + cells.view()[idx.second].eleid();
    }

  }  // namespace mappingtools

}  // namespace hgcal

#endif
