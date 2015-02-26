#include "RecoParticleFlow/PandoraTranslator/interface/steerManager.h"

steerManager:: steerManager()
{
}
steerManager:: steerManager(std::string _fname_)
{
   setFname(_fname_);
}
steerManager:: ~steerManager()
{
}
void steerManager:: setFname(std::string _fname_)
{
   _steerFileName = _fname_;
}

void steerManager:: addSingleParameter(std::string _singleParName_)
{
   singleParamList.push_back(_singleParName_);
   return;
}


void steerManager:: addArrayParameter(std::string _arrayParName_)
{
   arrayParamList.push_back(_arrayParName_);
   return;
}

bool steerManager:: read()
{
   //std::ifstream steerFile(_steerFileName, std::ifstream::in);
   std::ifstream steerFile(_steerFileName.c_str(), std::ifstream::in);

   if (!steerFile.is_open()) {
      std::cout << "steerManager:: steering file ("
         << _steerFileName << ") does not exit"
         << std::endl;
      return false;
   }

   std::cout << "reading steering file: " << _steerFileName << std::endl;

   while ( !steerFile.eof()) {
      std::string linebuf;
      getline( steerFile, linebuf );
      if (linebuf.substr(0,1) == "#") continue;
      if (linebuf.substr(0,2) == "//") continue;
      if (linebuf.empty()) continue;

      std::string paraName;

      std::stringstream ss(linebuf);

      ss >> paraName;

      //read single parameters ------------------
      std::list<std::string>::iterator it1;
      for (it1=singleParamList.begin(); it1!=singleParamList.end(); it1++) {
         std::string aParName = *it1;
         if (aParName == paraName) {
            ss >> _singleParas[paraName];
            std::cout << "read single: " 
               << paraName << ": " << _singleParas[paraName] << std::endl;
         }
      }

      //read array parameters ------------------
      std::list<std::string>::iterator it2;
      for (it2=arrayParamList.begin(); it2!=arrayParamList.end(); it2++) {
         std::string aParName = *it2;
         if (aParName == paraName) {
            std::cout << "read array [" << paraName << "]" << std::endl;
            double value;
            while (ss >> value) {
               _vectorParas[paraName].push_back(value);
               std::cout << value << " ";
            }
            std::cout << std::endl;
         }
      }
   }

   return true;
}

void steerManager:: printPars()
{
   std::cout << "steerManager:: print single parameters:" << std::endl;
   std::list<std::string>::iterator it1;
   for (it1=singleParamList.begin(); it1!=singleParamList.end(); it1++) {
      std::cout << *it1 << ": " << _singleParas[*it1] << std::endl;
   }

   std::cout << "steerManager:: print array parameters:" << std::endl;
   std::list<std::string>::iterator it2;
   for (it2=arrayParamList.begin(); it2!=arrayParamList.end(); it2++) {
      std::list<double> values = _vectorParas[*it2];
      std::cout << *it2 << ":";
      std::list<double>::iterator ival;
      for (ival=values.begin(); ival!=values.end(); ival++) {
         std::cout << " " << (*ival);
      }
      std::cout << std::endl;
   }
}


/*****************************************************************************/ 
double steerManager:: getCorrectionAtPoint(double _value_, std::string _edgesName_,
      std::string _corrArrayName_)
{
   double _corr = 1.;

   std::list<double> edges = _vectorParas[_edgesName_];  //edges, e.g. array of eta

   std::list<double> corrvalues = _vectorParas[_corrArrayName_]; 
   // corresponding correction factors

   std::list<double>::iterator itEdge = edges.begin();
   std::list<double>::iterator itCorr = corrvalues.begin();

   for ( ; itEdge!=edges.end(); itEdge++, itCorr++) {
      std::list<double>::iterator itEdgeNext = itEdge; itEdgeNext++;

      double loEdge = (*itEdge);
      double hiEdge = (*itEdgeNext);

      if (loEdge <= _value_ && _value_ <= hiEdge) {
         //std::cout << "val: " << _value_ << ", loEdge: " << loEdge
         //   << ", hiEdge: " << hiEdge << std::endl;
         _corr = (*itCorr);
      }

      if (itEdgeNext==edges.end()) {
         //std::cout << "breaking" << std::endl;
         break;
      }
   }

   return _corr;
}

/*****************************************************************************/ 

double *steerManager:: getArrayPara(std::string _parname_ )
{
   std::list<double> values = _vectorParas[_parname_];

   int nVal = values.size();
   //double output[nVal];
   double *output = new double[nVal];
   int nb=-1;
   for (std::list<double>::iterator ival=values.begin();
         ival!=values.end(); ival++) {
      nb++;
      output[nb] = *ival;
   }
   return output;
}

double steerManager:: getSinglePara(std::string _parname_ )
{
   return _singleParas[_parname_];
}
