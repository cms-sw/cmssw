#ifndef gem_readout_GEMslotContents_h
#define gem_readout_GEMslotContents_h

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

    class GEMslotContents {
      //struct is a class with all members public by default
    public:     
      GEMslotContents(const std::string& slotFile) {
        slotFile_ = slotFile;
        getSlotCfg();
      };
    private:
      uint16_t slot[24];
      bool isFileRead;
      std::string slotFile_;

       void initSlots() {
        for (int i = 0; i < 24; ++i)
          slot[i] = 0xfff;
        isFileRead = false;
        return;
      };

      void getSlotCfg() {
        std::ifstream ifile;
        std::string path = std::getenv("BUILD_HOME");
        path +="/gemdaq-testing/gemreadout/data/";
        path += slotFile_;
        ifile.open(path);
        
        if(!ifile.is_open()) {
          //std::cout << "[GEMslotContents]: The file: " << ifile << " is missing.\n" << std::endl;
          isFileRead = false;
          return;
        };        
        
        for (int row = 0; row < 3; row++) {
          std::string line;
          std::getline(ifile, line);
          std::istringstream iss(line);
          if ( !ifile.good() ) break;
          for (int col = 0; col < 8; col++) {
            std::string val;
            std::getline(iss,val,',');
            std::stringstream convertor(val);
            convertor >> std::hex >> slot[8*row+col];
          }
        }
        ifile.close();
        isFileRead = true;
      };

    public:
      /*
       *  Slot Index converter from Hex ChipID
       */
      int GEBslotIndex(const uint32_t& GEBChipID) {
        int indxslot = -1;
        //std::cout << "\nUsing slot file: " << slotFile_ << std::endl;
        for (int islot = 0; islot < 24; islot++) {
          if ( (GEBChipID & 0x0fff ) == slot[islot] ) indxslot = islot;
        }//end for slot
        
        return (indxslot);
      };
    }; // end class GEMslotContents

#endif
