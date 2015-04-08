#ifndef __STEERMAN__
#define __STEERMAN__

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <list>

class steerManager {
   public :
      steerManager();
      steerManager(std::string _fname_);
      ~steerManager();

      void setFname(std::string _fname_);
      void addSingleParameter(std::string _singleParName_);
      void addArrayParameter(std::string _arrayParName_);

      bool read();

      void printPars();

      double *getArrayPara(std::string _parname_ );
      double getSinglePara(std::string _parname_ );

      double getCorrectionAtPoint(double _value_, std::string _edgesName_,
            std::string _corrArrayName_);

   private :
      std::string _steerFileName;

      std::list < std::string > singleParamList;
      std::list < std::string > arrayParamList;

      std::map < std::string,std::list < double > > _vectorParas;
      std::map < std::string,double > _singleParas;
};
#endif
