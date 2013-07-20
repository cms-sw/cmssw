//
// F.Ratnikov (UMd), Oct 28, 2005
// $Id: HcalDbASCIIIO.cc,v 1.68 2012/11/12 20:40:40 dlange Exp $
//
#include <vector>
#include <string>
#include <cstdio>
#include <sstream>

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalDcsDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalText2DetIdConverter.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace {
  class DetIdLess {
  public:
    bool operator () (DetId fFirst, DetId fSecond) const {
      HcalGenericDetId first (fFirst);
      HcalGenericDetId second (fSecond);
      if (first.genericSubdet () != second.genericSubdet ()) return first.genericSubdet () < second.genericSubdet ();
      if (first.isHcalDetId ()) {
	HcalDetId f1 (first);
	HcalDetId s1 (second);
	return	f1.zside () != s1.zside () ? f1.zside () < s1.zside () :
	  f1.iphi () != s1.iphi () ? f1.iphi () < s1.iphi () :
	  f1.ietaAbs () != s1.ietaAbs () ? f1.ietaAbs () < s1.ietaAbs () :
	  f1.depth () < s1.depth ();
      }
      else {
	return first.rawId() < second.rawId();
      }
    }
  };
  class HcalElectronicsIdLess {
  public:
    bool operator () (HcalElectronicsId first, HcalElectronicsId second) const {
      return
	first.readoutVMECrateId () != second.readoutVMECrateId () ? first.readoutVMECrateId () < second.readoutVMECrateId () :
	first.htrSlot () != second.htrSlot () ? first.htrSlot () < second.htrSlot () :
	first.htrTopBottom () != second.htrTopBottom () ? first.htrTopBottom () < second.htrTopBottom () :
	first.fiberIndex () != second.fiberIndex () ? first.fiberIndex () < second.fiberIndex () :
	first.fiberChanId () < second.fiberChanId ();
    }
  };
}

// ------------------------------ some little helpers ------------------------------

std::vector <std::string> splitString (const std::string& fLine) {
  std::vector <std::string> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size (); i++) {
    if (fLine [i] == ' ' || i == fLine.size ()) {
      if (!empty) {
	std::string item (fLine, start, i-start);
	result.push_back (item);
	empty = true;
      }
      start = i+1;
    }
    else {
      if (empty) empty = false;
    }
  }
  return result;
}

std::vector <unsigned int> splitStringToIntByComma (const std::string& fLine) {
  std::vector <unsigned int> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size (); i++) {
    if (fLine [i] == ',' || i == fLine.size ()) {
      if (!empty) {
	std::string itemString (fLine, start, i-start);
	result.push_back (atoi (itemString.c_str()) );
	empty = true;
      }
      start = i+1;
    }
    else {
      if (empty) empty = false;
    }
  }
  return result;
}

std::vector <float> splitStringToFloatByComma (const std::string& fLine) {
  std::vector <float> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size (); i++) {
    if (fLine [i] == ',' || i == fLine.size ()) {
      if (!empty) {
	std::string itemString (fLine, start, i-start);
	result.push_back (atof (itemString.c_str()) );
	empty = true;
      }
      start = i+1;
    }
    else {
      if (empty) empty = false;
    }
  }
  return result;
}

std::vector <double> splitStringToDoubleByComma (const std::string& fLine) {
  std::vector <double> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size (); i++) {
    if (fLine [i] == ',' || i == fLine.size ()) {
      if (!empty) {
	std::string itemString (fLine, start, i-start);
	result.push_back (atof (itemString.c_str()) );
	empty = true;
      }
      start = i+1;
    }
    else {
      if (empty) empty = false;
    }
  }
  return result;
}

DetId HcalDbASCIIIO::getId (const std::vector <std::string> & items) {
  HcalText2DetIdConverter converter (items [3], items [0], items [1], items [2]);
  return converter.getId ();
}

void HcalDbASCIIIO::dumpId (std::ostream& fOutput, DetId id) {
  HcalText2DetIdConverter converter (id);
  char buffer [1024];
  sprintf (buffer, "  %15s %15s %15s %15s",
	   converter.getField1 ().c_str (), converter.getField2 ().c_str (), converter.getField3 ().c_str (),converter.getFlavor ().c_str ());  
  fOutput << buffer;
}

void HcalDbASCIIIO::dumpIdShort (std::ostream& fOutput, DetId id) {
  HcalText2DetIdConverter converter (id);
  char buffer [1024];
  sprintf (buffer, "  %5s %4s %4s %10s",
           converter.getField1 ().c_str (), converter.getField2 ().c_str (), converter.getField3 ().c_str (),converter.getFlavor ().c_str ());
  fOutput << buffer;
}


// ------------------------------ start templates ------------------------------

template<class T>
bool from_string(T& t, const std::string& s, std::ios_base& (*f)(std::ios_base&)) {
  std::istringstream iss(s);
  return !(iss >> f >> t).fail();
}

template <class T,class S> 
bool getHcalObject (std::istream& fInput, T* fObject, S* fCondObject) {
  if (!fObject) return false; //fObject = new T;
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    if (items.size () < 8) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 8 items: eta, phi, depth, subdet, 4x values" << std::endl;
      continue;
    }
    DetId id = HcalDbASCIIIO::getId (items);
    
//    if (fObject->exists(id) )
//      edm::LogWarning("Redefining Channel") << "line: " << buffer << "\n attempts to redefine data. Ignored" << std::endl;
//    else
//      {
	fCondObject = new S(id, atof (items [4].c_str()), atof (items [5].c_str()), 
			   atof (items [6].c_str()), atof (items [7].c_str()));
	fObject->addValues(*fCondObject);
	delete fCondObject;
	//      }
  }

  return true;
}

template <class T>
bool dumpHcalObject (std::ostream& fOutput, const T& fObject) {
  char buffer [1024];
  sprintf (buffer, "# %15s %15s %15s %15s %8s %8s %8s %8s %10s\n", "eta", "phi", "dep", "det", "cap0", "cap1", "cap2", "cap3", "DetId");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    const float* values = fObject.getValues (*channel)->getValues ();
    if (values) {
      HcalDbASCIIIO::dumpId (fOutput, *channel);
      sprintf (buffer, " %8.5f %8.5f %8.5f %8.5f %10X\n",
	       values[0], values[1], values[2], values[3], channel->rawId ());
      fOutput << buffer;
    }
  }
  return true;
}

template <class T,class S> 
bool getHcalSingleFloatObject (std::istream& fInput, T* fObject, S* fCondObject) {
  if (!fObject) return false; //fObject = new T;
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    if (items.size () < 5) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 5 items: eta, phi, depth, subdet, value" << std::endl;
      continue;
    }
    DetId id = HcalDbASCIIIO::getId (items);
    
//    if (fObject->exists(id) )
//      edm::LogWarning("Redefining Channel") << "line: " << buffer << "\n attempts to redefine data. Ignored" << std::endl;
//    else
//      {
	fCondObject = new S(id, atof (items [4].c_str()) );
	fObject->addValues(*fCondObject);
	delete fCondObject;
	//      }
  }
  return true;
}

template <class T>
bool dumpHcalSingleFloatObject (std::ostream& fOutput, const T& fObject) {
  char buffer [1024];
  sprintf (buffer, "# %15s %15s %15s %15s %8s %10s\n", "eta", "phi", "dep", "det", "value", "DetId");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    const float value = fObject.getValues (*channel)->getValue ();
    HcalDbASCIIIO::dumpId (fOutput, *channel);
    sprintf (buffer, " %8.5f %10X\n",
	     value, channel->rawId ());
    fOutput << buffer;
  }
  return true;
}

template <class T,class S> 
bool getHcalSingleIntObject (std::istream& fInput, T* fObject, S* fCondObject) {
  if (!fObject) return false; //fObject = new T;
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    if (items.size () < 5) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 5 items: eta, phi, depth, subdet, value" << std::endl;
      continue;
    }
    DetId id = HcalDbASCIIIO::getId (items);
    
//    if (fObject->exists(id) )
//      edm::LogWarning("Redefining Channel") << "line: " << buffer << "\n attempts to redefine data. Ignored" << std::endl;
//    else
//      {
	fCondObject = new S(id, atoi (items [4].c_str()) );
	fObject->addValues(*fCondObject);
	delete fCondObject;
	//      }
  }
  return true;
}

template <class T>
bool dumpHcalSingleIntObject (std::ostream& fOutput, const T& fObject) {
  char buffer [1024];
  sprintf (buffer, "# %15s %15s %15s %15s %8s %10s\n", "eta", "phi", "dep", "det", "value", "DetId");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    const int value = fObject.getValues (*channel)->getValue ();
    HcalDbASCIIIO::dumpId (fOutput, *channel);
    sprintf (buffer, " %15d %10X\n",
	     value, channel->rawId ());
    fOutput << buffer;
  }
  return true;
}

template <class T,class S>
bool getHcalMatrixObject (std::istream& fInput, T* fObject, S* fCondObject) {
  if (!fObject) return false; //fObject = new T;
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    DetId firstid = HcalDbASCIIIO::getId (items);
    fCondObject = new S(firstid.rawId());
    for(int j = 0; j != 10; j++) fCondObject->setValue(atoi(items[4].c_str()), 0, j, atof(items[j+5].c_str()));
    for(int i = 1; i != 40; i++){
       fInput.getline(buffer, 1024);
       items = splitString (std::string (buffer));
       DetId id = HcalDbASCIIIO::getId (items);
       if(id.rawId() != firstid.rawId()) break;//throw cms::Exception("Wrong number of elements");
       for(int j = 0; j != 10; j++) fCondObject->setValue(atoi(items[4].c_str()), i%10, j, atof(items[j+5].c_str()));
     }
     fObject->addValues(*fCondObject);
     delete fCondObject;
  }
  return true;
}

template <class T>
bool dumpHcalMatrixObject (std::ostream& fOutput, const T& fObject) {
  char buffer [1024];
  sprintf (buffer, "# %5s %5s %5s %5s %5s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %10s\n",
        "eta", "phi", "dep", "det", "capid","c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "DetId");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    float thisline[10];
    for(int m = 0; m != 4; m++){
       for(int i = 0; i != 10; i++){
          for(int j = 0; j != 10; j++){
//          std::cout <<"test "<<(fObject.getValues(*channel))->getValue(0,0,0);
            thisline[j] = fObject.getValues(*channel)->getValue(m,i,j);
//          thisline[j] = fObject.getValues(*channel)->getValue(1,1,1);
          }
          HcalDbASCIIIO::dumpId (fOutput, *channel);
          sprintf(buffer, " %5i %8.5f %8.5f %8.5f %8.5f  %8.5f %8.5f %8.5f %8.5f  %8.5f %8.5f %10X\n",
           m, thisline[0], thisline[1], thisline[2], thisline[3], thisline[4], thisline[5], thisline[6], thisline[7],
            thisline[8], thisline[9], channel->rawId());
          fOutput << buffer;
       }
    }
  }

  return true;
}

// ------------------------------ end templates ------------------------------

bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalGains* fObject) {return getHcalObject (fInput, fObject, new HcalGain);}
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalGains& fObject) {return dumpHcalObject (fOutput, fObject);}
bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalGainWidths* fObject) {return getHcalObject (fInput, fObject, new HcalGainWidth);}
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalGainWidths& fObject) {return dumpHcalObject (fOutput, fObject);}

bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalRespCorrs* fObject) {return getHcalSingleFloatObject (fInput, fObject, new HcalRespCorr); }
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalRespCorrs& fObject) {return dumpHcalSingleFloatObject (fOutput, fObject); }

bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalLUTCorrs* fObject) {return getHcalSingleFloatObject (fInput, fObject, new HcalLUTCorr); }
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalLUTCorrs& fObject) {return dumpHcalSingleFloatObject (fOutput, fObject); }

bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalPFCorrs* fObject) {return getHcalSingleFloatObject (fInput, fObject, new HcalPFCorr); }
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalPFCorrs& fObject) {return dumpHcalSingleFloatObject (fOutput, fObject); }

bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalTimeCorrs* fObject) {return getHcalSingleFloatObject (fInput, fObject, new HcalTimeCorr); }
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalTimeCorrs& fObject) {return dumpHcalSingleFloatObject (fOutput, fObject); }

bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalZSThresholds* fObject) {return getHcalSingleIntObject (fInput, fObject, new HcalZSThreshold); }
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalZSThresholds& fObject) {return dumpHcalSingleIntObject (fOutput, fObject); }

bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalValidationCorrs* fObject) {return getHcalSingleFloatObject (fInput, fObject, new HcalValidationCorr); }
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalValidationCorrs& fObject) {return dumpHcalSingleFloatObject (fOutput, fObject); }
bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalCholeskyMatrices* fObject) {return getHcalMatrixObject (fInput, fObject, new HcalCholeskyMatrix); }
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalCholeskyMatrices& fObject) {return dumpHcalMatrixObject (fOutput, fObject); }
bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalCovarianceMatrices* fObject) {return getHcalMatrixObject (fInput, fObject, new HcalCovarianceMatrix); }
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalCovarianceMatrices& fObject) {return dumpHcalMatrixObject (fOutput, fObject); }


// ------------------------------ start specific implementations ------------------------------
bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalRecoParams* fObject)
{
  if (!fObject) return false; // fObject = new HcalRecoParams();  This was always useless...
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    if (items.size () < 6) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 6 items: eta, phi, depth, subdet, param1, param2" << std::endl;
      continue;
    }
    DetId id = HcalDbASCIIIO::getId (items);

    int packingScheme =0;
    if(items.size ()>22) {
       packingScheme = atoi (items [22].c_str());
    }

    int param1=0;
    int param2=0;
    if(packingScheme==0) {
       param1=atoi (items [4].c_str());
       param2=atoi (items [5].c_str());
    }   // packing scheme 0  (old format).

    if(packingScheme==1) {
       //             0 1 2 3 4 5 6 7 8 9
       int aabits[6]= {1,1,  8, 4, 4,  9};
       int aamax[ 6]= {1,1,255,15,15,511};

       int bbbits[10]={1, 4,1, 4,1, 4, 4, 4, 4, 4};
       int bbmax [10]={1,15,1,15,1,15,15,15,15,15};

        // param 1
       int j=0;
       int aa;
       int aashift=0;
       for(int i=0; i<6; i++) {
          j=i+7;
          if(i==2) {
            float phase=atof (items [j].c_str());
            float xphase=(phase+32.0)*4.0;   // range of phase [-30.0,30.0] 
            aa=xphase;
          } else {
            aa=atoi (items [j].c_str());
          } 
          if(aa>aamax[i] || aa<0) {
              edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n value for a"<<i<<" should be less than"<<aamax[i]<< std::endl;
          }
          param1=param1|aa<<aashift;
          aashift=aashift+aabits[i];
       }

       // param 2
       int bb;
       int bbshift=0;
       for(int i=0; i<10; i++) {
          j=i+13;
          bb=atoi (items [j].c_str()); 
          if(bb>bbmax[i]) {
              edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n value for b"<<i<<" should be less than"<<bbmax[i]<< std::endl;
          }
          param2=param2|bb<<bbshift;
          bbshift=bbshift+bbbits[i];
       }
    } // packing sheme 1.    
   
    // HcalRecoParam* fCondObject = new HcalRecoParam(id, atoi (items [4].c_str()), atoi (items [5].c_str()) );

    // std::cout<<"  buffer "<<buffer<<std::endl;
    // std::cout<<"  param1, param2 "<<param1<<"  "<<param2<<std::endl;

    HcalRecoParam* fCondObject = new HcalRecoParam(id, param1, param2 );
    fObject->addValues(*fCondObject);
    delete fCondObject;
  }
  return true;
}
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalRecoParams& fObject)
{
  char buffer [1024];
  //  sprintf (buffer, "# %15s %15s %15s %15s %18s %15s %10s\n", "eta", "phi", "dep", "det", "firstSample", "samplesToAdd", "DetId");
  // fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  int  myCounter=0;
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    myCounter++;
    int param1=fObject.getValues (*channel)->param1();
    int param2=fObject.getValues (*channel)->param2();
    int packingScheme=fObject.getValues (*channel)->packingScheme();

    // std::cout<<"  Param1 "<<Param1<<"  Param2 "<<Param2<<"  packing "<<packingScheme<<std::endl;

    if(packingScheme==0) {
      // old format
      if(myCounter==1) {
           sprintf (buffer, "# %15s %15s %15s %15s %18s %15s %10s\n", "eta", "phi", "dep", "det", "firstSample", "samplesToAdd", "DetId");
      }
      HcalDbASCIIIO::dumpId(fOutput, *channel);
      sprintf (buffer, " %15d %15d %16X\n",
             fObject.getValues (*channel)->firstSample(), fObject.getValues (*channel)->samplesToAdd(), channel->rawId ());
      fOutput << buffer;
    }
   
    if(packingScheme==1) {

       if(myCounter==1) {
          char lineT[100],lineA[200],lineB[200];
          //
          sprintf (lineT, "#%50s","  ");   fOutput << lineT;
          sprintf (lineA, " %31s","a0: correctForPhaseContainment"); fOutput << lineA;
          sprintf (lineB, " %36s","b0: useLeakCorrection\n"); fOutput << lineB;
          //
          sprintf (lineT, "#%50s","  ");   fOutput << lineT;
          sprintf (lineA, " %31s","a1: correctForLeadingEdge"); fOutput << lineA;
          sprintf (lineB, " %36s","b1: leakCorrectionID\n"); fOutput << lineB;
          //
          sprintf (lineT, "#%50s","  ");   fOutput << lineT;
          sprintf (lineA, " %31s","a2: correctionPhaseNS"); fOutput << lineA;
          sprintf (lineB, " %36s","b2:  correctForTimeslew\n"); fOutput << lineB;
          //
          sprintf (lineT, "#%50s","  ");   fOutput << lineT;
          sprintf (lineA, " %31s","a3: firstSample"); fOutput << lineA;
          sprintf (lineB, " %36s","b3: timeslewCorrectionID\n"); fOutput << lineB;
          //
          sprintf (lineT, "#%50s","  ");   fOutput << lineT;
          sprintf (lineA, " %31s","a4: samplesToAdd"); fOutput << lineA;
          sprintf (lineB, " %36s","b4: correctTiming\n"); fOutput << lineB;
          //
          sprintf (lineT, "#%50s","  ");   fOutput << lineT;
          sprintf (lineA, " %31s","a5: pulseShapeID"); fOutput << lineA;
          sprintf (lineB, " %36s","b5: firstAuxTS\n"); fOutput << lineB;
          //
          sprintf (lineT, "#%50s","  ");   fOutput << lineT;
          sprintf (lineA, " %31s","  "); fOutput << lineA;
          sprintf (lineB, " %36s","b6: specialCaseID\n"); fOutput << lineB;
          //
          sprintf (lineT, "#%50s","  ");   fOutput << lineT;
          sprintf (lineA, " %31s","  "); fOutput << lineA;
          sprintf (lineB, " %36s","b7: noiseFlaggingID\n"); fOutput << lineB;
          //
         sprintf (lineT, "#%50s","  ");   fOutput << lineT;
         sprintf (lineA, " %31s","  "); fOutput << lineA;
         sprintf (lineB, " %36s","b8: pileupCleaningID\n"); fOutput << lineB;

         sprintf (lineT, "#%50s","  ");   fOutput << lineT;
         sprintf (lineA, " %31s","  "); fOutput << lineA;
         sprintf (lineB, " %36s","b9: packingScheme\n"); fOutput << lineB;

         //  
         sprintf (lineT, "# %5s %4s %4s %10s %11s %10s %10s", "eta", "phi", "dep", "det", "param1", "param2", "DetId");
         fOutput << lineT;

         sprintf (lineA, " %6s %4s %6s %4s %4s %4s", "a0", "a1", "a2", "a3", "a4", "a5");
         fOutput << lineA;

         sprintf (lineB, " %6s %3s %3s %3s %3s %3s %3s %3s %3s\n", "b0", "b1", "b2", "b3", "b4", "b5", "b6",  "b7", "b8");
         fOutput << lineB;
       }

       HcalDbASCIIIO::dumpIdShort(fOutput, *channel);
       sprintf (buffer, " %11d %10d %10X", param1, param2, channel->rawId ());
       fOutput << buffer;

       bool  aa0=fObject.getValues (*channel)->correctForPhaseContainment();
       bool  aa1=fObject.getValues (*channel)->correctForLeadingEdge();
       float aa2=fObject.getValues (*channel)->correctionPhaseNS();
       int   aa3=fObject.getValues (*channel)->firstSample();
       int   aa4=fObject.getValues (*channel)->samplesToAdd();
       int   aa5=fObject.getValues (*channel)->pulseShapeID();
       sprintf (buffer, " %6d %4d %6.1f %4d %4d %4d",aa0,aa1,aa2,aa3,aa4,aa5);
       fOutput << buffer;

       bool bb0=fObject.getValues (*channel)->useLeakCorrection();
       int  bb1=fObject.getValues (*channel)->leakCorrectionID();
       bool bb2=fObject.getValues (*channel)->correctForTimeslew();
       int  bb3=fObject.getValues (*channel)->timeslewCorrectionID();
       bool bb4=fObject.getValues (*channel)->correctTiming();
       int  bb5=fObject.getValues (*channel)->firstAuxTS();
       int  bb6=fObject.getValues (*channel)->specialCaseID();
       int  bb7=fObject.getValues (*channel)->noiseFlaggingID();
       int  bb8=fObject.getValues (*channel)->pileupCleaningID();
       int  bb9=fObject.getValues (*channel)->packingScheme();
       sprintf(buffer," %6d %3d %3d %3d %3d %3d %3d %3d %3d %3d\n",bb0,bb1,bb2,bb3,bb4,bb5,bb6,bb7,bb8,bb9);
       fOutput << buffer;
     }   // packingScheme 1.

  }  // loop ever channels
  return true;
}

bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalLongRecoParams* fObject)
{
  if (!fObject) return false; // fObject = new HcalLongRecoParams();
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    if (items.size() < 5) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 6 items: eta, phi, depth, subdet, signalTSs, noiseTSs" << std::endl;
      continue;
    }
    if (items.size() > 7) {
      edm::LogWarning("Format Problem ?") << "Check line: " << buffer << "\n line must contain 6 items: eta, phi, depth, subdet, signalTSs, noiseTSs. "
					  << "\n ! signalTS and noiseTS must be of format <ts1,ts2,ts3,...> withOUT spaces. Ignoring line for safety" << std::endl;
      continue;
    }
    DetId id = HcalDbASCIIIO::getId (items);
    
    std::vector<unsigned int> mySignalTS = splitStringToIntByComma(items[4]);
    std::vector<unsigned int> myNoiseTS = splitStringToIntByComma(items[5]);

    HcalLongRecoParam* fCondObject = new HcalLongRecoParam(id, mySignalTS, myNoiseTS );
    fObject->addValues(*fCondObject);
    delete fCondObject;
  }
  return true;
}
bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalTimingParams* fObject)
{
  if (!fObject) return false; // fObject = new HcalTimingParams();
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    if (items.size () < 7) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 7 items: eta, phi, depth, subdet, nhit, phase, rms,detid" << std::endl;
      continue;
    }
    //std::cout<<"items[3] "<<items [3]<<std::endl;
    //std::cout<<"items[0] "<<items [0]<<std::endl;
    //std::cout<<"items[1] "<<items [1]<<std::endl;
    //std::cout<<"items[2] "<<items [2]<<std::endl;

    //std::cout<<"items[4] "<<items [4]<<std::endl;
    //std::cout<<"items[5] "<<items [5]<<std::endl;
    //std::cout<<"items[6] "<<items [6]<<std::endl;
    DetId id = HcalDbASCIIIO::getId (items);
    //std::cout<<"calculated id "<<id.rawId()<<std::endl;
    HcalTimingParam* fCondObject = new HcalTimingParam(id, atoi (items [4].c_str()), atof (items [5].c_str()), atof (items [6].c_str()));
    fObject->addValues(*fCondObject);
    delete fCondObject;
  }
  return true;
}
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalTimingParams& fObject)
{
  char buffer [1024];
  sprintf (buffer, "# %15s %15s %15s %15s %15s %15s %15s %15s\n", "eta", "phi", "dep", "det", "nhit", "mean","rms" ,"DetId");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    HcalDbASCIIIO::dumpId (fOutput, *channel);
    sprintf (buffer, " %15d %8.5f %8.5f %16X\n",
	     fObject.getValues (*channel)->nhits(), fObject.getValues (*channel)->phase(),fObject.getValues(*channel)->rms(),channel->rawId ());
    fOutput << buffer;
  }
  return true;
}

bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalLongRecoParams& fObject)
{
  char buffer [1024];
  sprintf (buffer, "# %15s %15s %15s %15s %10s %10s %10s\n", "eta", "phi", "dep", "det", "signalTSs", "noiseTSs", "DetId");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    HcalGenericDetId fId(*channel);
    if (fId.isHcalZDCDetId())
      {
	std::vector<unsigned int> vSignalTS = fObject.getValues (*channel)->signalTS();
	std::vector<unsigned int> vNoiseTS = fObject.getValues (*channel)->noiseTS();
	HcalDbASCIIIO::dumpId (fOutput, *channel);
	sprintf (buffer, "    ");
	fOutput << buffer;
	for (unsigned int i=0; i<vSignalTS.size(); i++)
	  {
	    if (i>0) {sprintf (buffer, ",");   fOutput << buffer;}
	    sprintf (buffer, "%u", vSignalTS.at(i));
	    fOutput << buffer;
	  }
	sprintf (buffer, "       ");
	fOutput << buffer;
	for (unsigned int i=0; i<vNoiseTS.size(); i++)
	  {
	    if (i>0) { sprintf (buffer, ",");   fOutput << buffer;}
	    sprintf (buffer, "%u", vNoiseTS.at(i));
	    fOutput << buffer;
	  }
	sprintf (buffer, "     %10X\n", channel->rawId ());
	fOutput << buffer;
      }
  }
  return true;
}

bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalMCParams* fObject)
{
  if (!fObject) return false; // fObject = new HcalMCParams();
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    if (items.size () < 5) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 5 items: eta, phi, depth, subdet, signalShape" << std::endl;
      continue;
    }
    DetId id = HcalDbASCIIIO::getId (items);

    int packingScheme=0;
    if(items.size ()>11) {
       packingScheme = atoi (items [11].c_str());
    }

    int param1=0;
    if(packingScheme==0) {
        param1=atoi (items [4].c_str());
    }

    if(packingScheme==1) {
       int aabits[6]={  9,1, 4,  8,5, 4};   // 4 spear bits added to aabits[5]
       int aamax [6]={511,1,15,255,1,16};
       int j=0;
       int aa;
       int aashift=0;
       for(int i=0; i<6; i++) {
          j=i+6;
          if(i==3) {
            float phase=atof (items [j].c_str());
            float xphase=(phase+32.0)*4.0;   // range of phase [-30.0,30.0] 
            aa=xphase;
          } else {
            aa=atoi (items [j].c_str());
          }
          if(aa>aamax[i] || aa<0) {
              edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n value for a"<<i<<" should be less than"<<aamax[i]<< std::endl;
          }

          param1=param1|aa<<aashift;
          aashift=aashift+aabits[i];
       }
    }

    HcalMCParam* fCondObject = new HcalMCParam(id, param1 );
    fObject->addValues(*fCondObject);
    delete fCondObject;
  }
  return true;
}
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalMCParams& fObject)
{
  char buffer [1024];
  // sprintf (buffer, "# %15s %15s %15s %15s %14s %10s\n", "eta", "phi", "dep", "det", "signalShape", "DetId");
  // fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  int  myCounter=0;
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    myCounter++;;  
    int packingScheme=fObject.getValues (*channel)->packingScheme();
    if(packingScheme==0) {
       if(myCounter==1) {
          sprintf (buffer, "# %15s %15s %15s %15s %14s %10s\n", "eta", "phi", "dep", "det", "signalShape", "DetId");
          fOutput << buffer;
       }
           const int value = fObject.getValues (*channel)->signalShape();
       HcalDbASCIIIO::dumpId (fOutput, *channel);
       sprintf (buffer, " %10d %17X\n", value, channel->rawId ());
       fOutput << buffer;
    }  // packingScheme 0
    if(packingScheme==1) {
      if(myCounter==1) {
        char lineT[100],lineA[200];
        //
        sprintf (lineT, "#%40s","  ");   fOutput << lineT;
        sprintf (lineA, " %31s","a0: signalShape\n"); fOutput << lineA;
        sprintf (lineT, "#%40s","  ");   fOutput << lineT;
        sprintf (lineA, " %31s","a1: syncPhase\n"); fOutput << lineA;
        sprintf (lineT, "#%40s","  ");   fOutput << lineT;
        sprintf (lineA, " %31s","a2: binOfMaximum\n"); fOutput << lineA;
        sprintf (lineT, "#%40s","  ");   fOutput << lineT;
        sprintf (lineA, " %31s","a3: timePhase\n"); fOutput << lineA;
        sprintf (lineT, "#%40s","  ");   fOutput << lineT;
        sprintf (lineA, " %31s","a4: timeSmearing\n"); fOutput << lineA;
        sprintf (lineT, "#%40s","  ");   fOutput << lineT;
        sprintf (lineA, " %31s","a5: packingScheme\n"); fOutput << lineA;
        sprintf (lineT, "# %5s %4s %4s %10s %11s %10s", "eta", "phi", "dep", "det", "param1", "DetId");
        fOutput << lineT;
        sprintf (lineA, " %6s %4s %4s %6s %4s %4s\n", "a0", "a1", "a2", "a3", "a4", "a5");
        fOutput << lineA;
      }
      int   param1 = fObject.getValues (*channel)->param1();
      HcalDbASCIIIO::dumpIdShort (fOutput, *channel);
      sprintf (buffer, " %11d  %10X", param1, channel->rawId ());
      fOutput << buffer;
     int   aa0 = fObject.getValues (*channel)->signalShape(); 
     bool  aa1 = fObject.getValues (*channel)->syncPhase(); 
     int   aa2 = fObject.getValues (*channel)->binOfMaximum();
     float aa3 = fObject.getValues (*channel)->timePhase();
     bool  aa4 = fObject.getValues (*channel)->timeSmearing() ;
     int   aa5 = fObject.getValues (*channel)->packingScheme();
     sprintf (buffer, "%6d %4d %4d %6.1f %4d %4d\n",aa0,aa1,aa2,aa3,aa4,aa5);
     fOutput << buffer;
    }
  }
  return true;
}

bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalPedestals* fObject) {
  char buffer [1024];

  while (fInput.getline(buffer, 1024)) {
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    else {
      if (items[0] == "#U")
	{
	  if (items[1] == (std::string)"ADC") fObject->setUnitADC(true);
	    else if (items[1] == (std::string)"fC") fObject->setUnitADC(false);
	  else 
	    {
	      edm::LogWarning("Pedestal Unit Error") << "Unrecognized unit for pedestals. Assuming fC." << std::endl;
	      fObject->setUnitADC(false);
	    }
	  break;
	}
      else
	{
	  edm::LogWarning("Pedestal Unit Missing") << "The unit for the pedestals is missing in the txt file." << std::endl;
	  return false;
	}
    }
  }
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue;
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    if (items.size () < 8) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 8 items: eta, phi, depth, subdet, 4x values" 
				      << " or 12 items: eta, phi, depth, subdet, 4x values for mean, 4x values for width"
				      << std::endl;
      continue;
    }
    DetId id = getId (items);
    
//    if (fObject->exists(id) )
//      edm::LogWarning("Redefining Channel") << "line: " << buffer << "\n attempts to redefine data. Ignored" << std::endl;
//    else
//      {

    if (items.size() < 12) // old format without widths
      {
	HcalPedestal* fCondObject = new HcalPedestal(id, atof (items [4].c_str()), atof (items [5].c_str()), 
						     atof (items [6].c_str()), atof (items [7].c_str()), 
						     0., 0., 0., 0. );
	fObject->addValues(*fCondObject);
	delete fCondObject;
      }
    else // new format with widths
      {
	HcalPedestal* fCondObject = new HcalPedestal(id, atof (items [4].c_str()), atof (items [5].c_str()), 
						     atof (items [6].c_str()), atof (items [7].c_str()), 
						     atof (items [8].c_str()), atof (items [9].c_str()),
						     atof (items [10].c_str()), atof (items [11].c_str()) );
	fObject->addValues(*fCondObject);
	delete fCondObject;
      }

	//      }
  }
  return true;
}


bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalPedestals& fObject) {
  char buffer [1024];
  if (fObject.isADC() ) sprintf (buffer, "#U ADC  << this is the unit \n");
  else  sprintf (buffer, "#U fC  << this is the unit \n");
  fOutput << buffer;

  sprintf (buffer, "# %15s %15s %15s %15s %8s %8s %8s %8s %8s %8s %8s %8s %10s\n", "eta", "phi", "dep", "det", "cap0", "cap1", "cap2", "cap3", "widthcap0", "widthcap1", "widthcap2", "widthcap3", "DetId");
  fOutput << buffer;

  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    const float* values = fObject.getValues (*channel)->getValues ();
    if (values) {
      dumpId (fOutput, *channel);
      sprintf (buffer, " %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %10X\n",
	       values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], channel->rawId ());
      fOutput << buffer;
    }
  }
  return true;
}


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalChannelQuality* fObject) 
{
  if (!fObject) return false; //fObject = new HcalChannelQuality;
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    if (items.size () < 6) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 6 items: eta, phi, depth, subdet, base - either (hex) or (dec), value" << std::endl;
      continue;
    }
    DetId id = getId (items);
    
//    if (fObject->exists(id) )
//      edm::LogWarning("Redefining Channel") << "line: " << buffer << "\n attempts to redefine data. Ignored" << std::endl;
//    else
//      {
    uint32_t mystatus;
    if (items[4] == "(hex)")
      sscanf(items[5].c_str(),"%X", &mystatus);
    else if (items[4] == "(dec)")
      sscanf(items[5].c_str(),"%u", &mystatus);
    else
      {
	edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n value field must contain the base: one of (hex), (dec)" << std::endl;
	continue;
      }

    HcalChannelStatus* fCondObject = new HcalChannelStatus(id, mystatus); //atoi (items [4].c_str()) );
    fObject->addValues(*fCondObject);
    delete fCondObject;
	//      }
  }
  return true;
}


bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalChannelQuality& fObject) {
  char buffer [1024];
  sprintf (buffer, "# %15s %15s %15s %15s %15s %10s\n", "eta", "phi", "dep", "det", "(base) value", "DetId");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    const int value = fObject.getValues (*channel)->getValue ();
    dumpId (fOutput, *channel);
    sprintf (buffer, "%6s %15X %10X\n", "(hex)",
	     value, channel->rawId ());
    fOutput << buffer;
  }
  return true;
}


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalL1TriggerObjects* fObject)
{
  char buffer [1024];

  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') 
      {
	if (buffer [1] == 'T') // contains tag name
	  {
	    std::vector <std::string> items = splitString (std::string (buffer) );
	    fObject->setTagString(items[1]);
	    continue;
	  }
	if (buffer [1] == 'A') // contains algo name
	  {
	    std::vector <std::string> items = splitString (std::string (buffer) );
	    fObject->setAlgoString(items[1]);
	    continue;
	  }
	else continue; //ignore comment
      }
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    if (items.size () < 7) { 
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 7 items: eta, phi, depth, subdet, pedestal, resp.corr.gain, flag" << std::endl;
      continue;
    }
    DetId id = getId (items);
    
    HcalL1TriggerObject* fCondObject = new HcalL1TriggerObject(id, atof(items[4].c_str()), atof(items[5].c_str()), atoi(items[6].c_str()) );
    
    fObject->addValues(*fCondObject);
    delete fCondObject;
  }
  return true;
}

bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalL1TriggerObjects& fObject)
{
  char buffer [1024];
  //first print tag and algo
  sprintf (buffer, "#T %s  << this is the tag name \n", fObject.getTagString().c_str() );
  fOutput << buffer;
  sprintf (buffer, "#A %s  << this is the algorithm name \n", fObject.getAlgoString().c_str() );
  fOutput << buffer;

  //then the values
  sprintf (buffer, "# %15s %15s %15s %15s %8s %13s %8s %10s\n", 
	   "eta", "phi", "dep", "det", "ped", "respcorrgain", "flag",
	   "DetId");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  //  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    const HcalL1TriggerObject* item = fObject.getValues (*channel);
    if (item) {
      dumpId (fOutput, *channel);
      sprintf (buffer, " %10.7f %10.7f %12d %10X\n",
	       item->getPedestal(), item->getRespGain(), item->getFlag(), channel->rawId ());
      fOutput << buffer;
    }
  }
  return true;

}


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalPedestalWidths* fObject) {
  char buffer [1024];
  int linecounter = 0;

  while (fInput.getline(buffer, 1024)) {
    linecounter++;
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    else {
      if (items[0] == (std::string)"#U")
	{
	  if (items[1] == (std::string)"ADC") fObject->setUnitADC(true); 
	  else if (items[1] == (std::string)"fC") fObject->setUnitADC(false);
	  else 
	    {
	      edm::LogWarning("Pedestal Width Unit Error") << "Unrecognized unit for pedestal widths. Assuming fC." << std::endl;
	      fObject->setUnitADC(false);
	    }
	  break;
	}
      else
	{
	  edm::LogWarning("Pedestal Width Unit Missing") << "The unit for the pedestal widths is missing in the txt file." << std::endl;
	  return false;
	}
    }
  }

  while (fInput.getline(buffer, 1024)) {
    linecounter++;
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    if (items.size () < 14) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line number: " << linecounter << "\n line must contain 14 items: eta, phi, depth, subdet, 10x correlations" 
				      << " or 20 items: eta, phi, depth, subdet, 16x correlations" 
				      << std::endl;
      continue;
    }
    DetId id = getId (items);

//    if (fObject->exists(id) )
//      edm::LogWarning("Redefining Channel") << "line: " << buffer << "\n attempts to redefine data. Ignored" << std::endl;
//    else
//      {

    if (items.size() < 20) //old format
      {
	HcalPedestalWidth values(id);
	values.setSigma (0, 0, atof (items [4].c_str()));
	values.setSigma (1, 0, atof (items [5].c_str()));
	values.setSigma (1, 1, atof (items [6].c_str()));
	values.setSigma (2, 0, atof (items [7].c_str()));
	values.setSigma (2, 1, atof (items [8].c_str()));
	values.setSigma (2, 2, atof (items [9].c_str()));
	values.setSigma (3, 0, atof (items [10].c_str()));
	values.setSigma (3, 1, atof (items [11].c_str()));
	values.setSigma (3, 2, atof (items [12].c_str()));
	values.setSigma (3, 3, atof (items [13].c_str()));
	values.setSigma (0, 1, 0.);
	values.setSigma (0, 2, 0.);
	values.setSigma (0, 3, 0.);
	values.setSigma (1, 2, 0.);
	values.setSigma (1, 3, 0.);
	values.setSigma (2, 3, 0.);
	fObject->addValues(values);	
      }
    else // new format
      {
	HcalPedestalWidth values(id);
	values.setSigma (0, 0, atof (items [4].c_str()) );
	values.setSigma (0, 1, atof (items [5].c_str()) );
	values.setSigma (0, 2, atof (items [6].c_str()) );
	values.setSigma (0, 3, atof (items [7].c_str()) );
	values.setSigma (1, 0, atof (items [8].c_str()) );
	values.setSigma (1, 1, atof (items [9].c_str()) );
	values.setSigma (1, 2, atof (items [10].c_str()) );
	values.setSigma (1, 3, atof (items [11].c_str()) );
	values.setSigma (2, 0, atof (items [12].c_str()) );
	values.setSigma (2, 1, atof (items [13].c_str()) );
	values.setSigma (2, 2, atof (items [14].c_str()) );
	values.setSigma (2, 3, atof (items [15].c_str()) );
	values.setSigma (3, 0, atof (items [16].c_str()) );
	values.setSigma (3, 1, atof (items [17].c_str()) );
	values.setSigma (3, 2, atof (items [18].c_str()) );
	values.setSigma (3, 3, atof (items [19].c_str()) );
	fObject->addValues(values);	
      }

	//      }
  }
  return true;
}

bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalPedestalWidths& fObject) {
  char buffer [1024];
  if (fObject.isADC() ) sprintf (buffer, "#U ADC  << this is the unit \n");
  else  sprintf (buffer, "#U fC  << this is the unit \n");
  fOutput << buffer;

  sprintf (buffer, "# %15s %15s %15s %15s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %10s\n", 
	   "eta", "phi", "dep", "det", 
	   "cov_0_0", "cov_0_1", "cov_0_2", "cov_0_3", "cov_1_0", "cov_1_1", "cov_1_2", "cov_1_3", "cov_2_0", "cov_2_1", "cov_2_2", "cov_2_3", "cov_3_0", "cov_3_1", "cov_3_2", "cov_3_3", 
	   "DetId");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    const HcalPedestalWidth* item = fObject.getValues (*channel);
    if (item) {
      dumpId (fOutput, *channel);
      sprintf (buffer, " %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %10X\n",
	       item->getSigma (0,0), item->getSigma (0,1), item->getSigma (0,2), item->getSigma (0,3), 
	       item->getSigma (1,0), item->getSigma (1,1), item->getSigma (1,2), item->getSigma (1,3),
	       item->getSigma (2,0), item->getSigma (2,1), item->getSigma (2,2), item->getSigma (2,3),
	       item->getSigma (3,0), item->getSigma (3,1), item->getSigma (3,2), item->getSigma (3,3), channel->rawId ());
      fOutput << buffer;
    }
  }
  return true;
}


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalQIEData* fObject) {
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()<1) continue;
    if (items [0] == "SHAPE") { // basic shape
      //this shape keyword is obsolete
    }
    else { // QIE parameters
      if (items.size () < 36) {
	edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 36 items: eta, phi, depth, subdet, 4 capId x 4 Ranges x offsets, 4 capId x 4 Ranges x slopes" << std::endl;
	continue;
      }
      DetId id = getId (items);
      fObject->sort ();
      //      try {
      //      fObject->getCoder (id);
      //      edm::LogWarning("Redefining Channel") << "line: " << buffer << "\n attempts to redefine data. Ignored" << std::endl;
	//      }
//      catch (cms::Exception& e) {
	HcalQIECoder coder (id.rawId ());
	int index = 4;
	for (unsigned capid = 0; capid < 4; capid++) {
	  for (unsigned range = 0; range < 4; range++) {
	    coder.setOffset (capid, range, atof (items [index++].c_str ()));
	  }
	}
	for (unsigned capid = 0; capid < 4; capid++) {
	  for (unsigned range = 0; range < 4; range++) {
	    coder.setSlope (capid, range, atof (items [index++].c_str ()));
	  }
	}
	if (items.size()>36)
	  coder.setQIEIndex(atoi(items[index++].c_str()));

	fObject->addCoder (coder);
//      }
    }
  }
  fObject->sort ();
  return true;
}

bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalQIEData& fObject) {
  std::cout <<"dumping object\n";
  char buffer [1024];

  fOutput << "# QIE data" << std::endl;
  sprintf (buffer, "# %15s %15s %15s %15s %36s %36s %36s %36s %36s %36s %36s %36s\n", 
	   "eta", "phi", "dep", "det", 
	   "4 x offsets cap0", "4 x offsets cap1", "4 x offsets cap2", "4 x offsets cap3",
	   "4 x slopes cap0", "4 x slopes cap1", "4 x slopes cap2", "4 x slopes cap3");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    const HcalQIECoder* coder = fObject.getCoder (*channel);
    dumpId (fOutput, *channel);
    for (unsigned capid = 0; capid < 4; capid++) {
      for (unsigned range = 0; range < 4; range++) {
	sprintf (buffer, " %8.5f", coder->offset (capid, range));
	fOutput << buffer;
      }
    }
    for (unsigned capid = 0; capid < 4; capid++) {
      for (unsigned range = 0; range < 4; range++) {
	sprintf (buffer, " %8.5f", coder->slope (capid, range));
	fOutput << buffer;
      }
    }
    sprintf (buffer, " %2d", coder->qieIndex());
    fOutput << buffer;
    fOutput << std::endl;
  }
  return true;
}

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalCalibrationQIEData* fObject) {
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size () < 36) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 36 items: eta, phi, depth, subdet, 32 bin values" << std::endl;
      continue;
    }
    DetId id = getId (items);
    fObject->sort ();
    //    try {
    //    fObject->getCoder (id);
    //    edm::LogWarning("Redefining Channel") << "line: " << buffer << "\n attempts to redefine data. Ignored" << std::endl;
      //    }
//    catch (cms::Exception& e) {
      HcalCalibrationQIECoder coder (id.rawId ());
      int index = 4;
      float values [32];
      for (unsigned bin = 0; bin < 32; bin++) {
	values[bin] = atof (items [index++].c_str ());
      }
      coder.setMinCharges (values);
      fObject->addCoder (coder);
//    }
  }
  fObject->sort ();
  return true;
}

bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalCalibrationQIEData& fObject) {
  char buffer [1024];
  fOutput << "# QIE data in calibration mode" << std::endl;
  sprintf (buffer, "# %15s %15s %15s %15s %288s\n", 
	   "eta", "phi", "dep", "det", "32 x charges");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    const HcalCalibrationQIECoder* coder = fObject.getCoder (*channel);
    if (coder) {
      dumpId (fOutput, *channel);
      const float* lowEdge = coder->minCharges ();
      for (unsigned bin = 0; bin < 32; bin++) {
	sprintf (buffer, " %8.5f", lowEdge [bin]);
	fOutput << buffer;
      }
      fOutput << std::endl;
    }
  }
  return true;
}


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalElectronicsMap* fObject) {
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size () < 12) {
      if (items.size()==0) continue; // no warning here
      if (items.size()<9) {
	edm::LogError("MapFormat") << "HcalElectronicsMap-> line too short: " << buffer;
	continue;
      }
      if (items[8]=="NA" || items[8]=="NT") {
	while (items.size()<12) items.push_back(""); // don't worry here
      } else if (items[8]=="HT") {
	if (items.size()==11) items.push_back("");
	else {
	  edm::LogError("MapFormat") << "HcalElectronicsMap-> Bad line: " << buffer 
				     << "\n HT line must contain at least 11 items: i  cr sl tb dcc spigot fiber fiberchan subdet=HT ieta iphi";
	  continue;
	}
      } else {
	edm::LogError("MapFormat") << "HcalElectronicsMap-> Bad line: " << buffer 
				   << "\n line must contain 12 items: i  cr sl tb dcc spigot fiber fiberchan subdet ieta iphi depth";
	continue;
      }
    }
    //    std::cout << "HcalElectronicsMap-> processing line: " << buffer << std::endl;
    int crate = atoi (items [1].c_str());
    int slot = atoi (items [2].c_str());
    int top = 1;
    if (items [3] == "b") top = 0;
    int dcc = atoi (items [4].c_str());
    int spigot = atoi (items [5].c_str());
    HcalElectronicsId elId;
    if (items[8] == "HT" || items[8] == "NT") {
      int slb = atoi (items [6].c_str());
      int slbCh = atoi (items [7].c_str());
      elId=HcalElectronicsId(slbCh, slb, spigot, dcc,crate,slot,top);
    } else {
      int fiber = atoi (items [6].c_str());
      int fiberCh = atoi (items [7].c_str());

      elId=HcalElectronicsId(fiberCh, fiber, spigot, dcc);
      elId.setHTR (crate, slot, top);
    }

    // first, handle undefined cases
    if (items [8] == "NA") { // undefined channel
      fObject->mapEId2chId (elId, DetId (HcalDetId::Undefined));
    } else if (items [8] == "NT") { // undefined trigger channel
      fObject->mapEId2tId (elId, DetId (HcalTrigTowerDetId::Undefined));
    } else {
      HcalText2DetIdConverter converter (items [8], items [9], items [10], items [11]);
      if (converter.isHcalDetId ()) { 
	fObject->mapEId2chId (elId, converter.getId ());
      }
      else if (converter.isHcalTrigTowerDetId ()) {
	fObject->mapEId2tId (elId, converter.getId ());
      }
      else if (converter.isHcalCalibDetId ()) {
	fObject->mapEId2chId (elId, converter.getId ());
      }
      else if (converter.isHcalZDCDetId ()) {
	fObject->mapEId2chId (elId, converter.getId ());
      }
      else {
	edm::LogWarning("Format Error") << "HcalElectronicsMap-> Unknown subdetector: " 
		  << items [8] << '/' << items [9] << '/' << items [10] << '/' << items [11] << std::endl; 
      }
    }
  }
  fObject->sort ();
  return true;
}

bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalElectronicsMap& fObject) {
  std::vector<HcalElectronicsId> eids = fObject.allElectronicsId ();
  char buf [1024];
  // changes by Jared, 6.03.09/(included 25.03.09)
  //  sprintf (buf, "#%10s %6s %6s %6s %6s %6s %6s %6s %15s %15s %15s %15s",
  sprintf (buf, "# %7s %3s %3s %3s %4s %7s %10s %14s %7s %5s %5s %6s",
	   "i", "cr", "sl", "tb", "dcc", "spigot", "fiber/slb", "fibcha/slbcha", "subdet", "ieta", "iphi", "depth");
  fOutput << buf << std::endl;

  for (unsigned i = 0; i < eids.size (); i++) {
    HcalElectronicsId eid = eids[i];
    if (eid.isTriggerChainId()) {
      DetId trigger = fObject.lookupTrigger (eid);
      if (trigger.rawId ()) {
	HcalText2DetIdConverter converter (trigger);
	// changes by Jared, 6.03.09/(included 25.03.09)
	//	sprintf (buf, " %10X %6d %6d %6c %6d %6d %6d %6d %15s %15s %15s %15s",
	sprintf (buf, " %7X %3d %3d %3c %4d %7d %10d %14d %7s %5s %5s %6s",
		 //		 i,
		 converter.getId().rawId(),
		 // changes by Jared, 6.03.09/(included 25.03.09)
		 //		 eid.readoutVMECrateId(), eid.htrSlot(), eid.htrTopBottom()>0?'t':'b', eid.dccid(), eid.spigot(), eid.fiberIndex(), eid.fiberChanId(),
		 eid.readoutVMECrateId(), eid.htrSlot(), eid.htrTopBottom()>0?'t':'b', eid.dccid(), eid.spigot(), eid.slbSiteNumber(), eid.slbChannelIndex(),
		 converter.getFlavor ().c_str (), converter.getField1 ().c_str (), converter.getField2 ().c_str (), converter.getField3 ().c_str ()
		 );
	fOutput << buf << std::endl;
      }
    } else {
      DetId channel = fObject.lookup (eid);
      if (channel.rawId()) {
	HcalText2DetIdConverter converter (channel);
	// changes by Jared, 6.03.09/(included 25.03.09)
	//	sprintf (buf, " %10X %6d %6d %6c %6d %6d %6d %6d %15s %15s %15s %15s",
	sprintf (buf, " %7X %3d %3d %3c %4d %7d %10d %14d %7s %5s %5s %6s",
		 //		 i,
		 converter.getId().rawId(),
		 eid.readoutVMECrateId(), eid.htrSlot(), eid.htrTopBottom()>0?'t':'b', eid.dccid(), eid.spigot(), eid.fiberIndex(), eid.fiberChanId(),
		 converter.getFlavor ().c_str (), converter.getField1 ().c_str (), converter.getField2 ().c_str (), converter.getField3 ().c_str ()
	       );
	fOutput << buf << std::endl;
      }
    }
  }
  return true;
}


bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalLutMetadata* fObject){
  if (!fObject) return false; //fObject = new HcalLutMetadata;
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    // now get non-channel data
    if (items.size() > 1 && 
	items[0].find("RctLsb")!=std::string::npos){
      fObject->setRctLsb( atof( items[1].c_str() ) );
      continue;
    }
    if (items.size() > 1 && 
	items[0].find("Gain")!=std::string::npos){
      fObject->setNominalGain( atof( items[1].c_str() ) );
      continue;
    }
    // now proceeed to per-channel data
    if (items.size () < 7) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 7 items: eta, phi, depth, subdet, Rcalib, LutGranularity, OutputLutThreshold" << std::endl;
      continue;
    }
    DetId id = getId (items);
    
    HcalLutMetadatum * fCondObject = new HcalLutMetadatum(id,
							  atof (items [4].c_str()),
							  atoi (items [5].c_str()),
							  atoi (items [6].c_str()));
    fObject->addValues(*fCondObject);
    delete fCondObject;
  }
  return true;
}


bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalLutMetadata& fObject){
  char buffer [1024];
  const float _rctLsb = fObject.getRctLsb();
  const float _gain   = fObject.getNominalGain();
  sprintf (buffer, "# %20s\n", "Non-channel data");
  fOutput << buffer;
  sprintf (buffer, "%8s %8.5f\n", "RctLsb", _rctLsb);
  fOutput << buffer;
  sprintf (buffer, "%8s %8.5f\n", "Gain", _gain);
  fOutput << buffer;
  sprintf (buffer, "# %15s %15s %15s %15s %8s %15s %19s %10s\n", "eta", "phi", "dep", "det", "Rcalib", "LutGranularity", "OutputLutThreshold", "DetId");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    const float   _rCalib             = fObject.getValues (*channel)->getRCalib();
    const uint8_t _lutGranularity     = fObject.getValues (*channel)->getLutGranularity();
    const uint8_t _outputLutThreshold = fObject.getValues (*channel)->getOutputLutThreshold();
    dumpId (fOutput, *channel);
    sprintf (buffer, " %8.5f %15d %19d %10X\n",
	     _rCalib,
	     _lutGranularity,
	     _outputLutThreshold,
	     channel->rawId ());
    fOutput << buffer;
  }
  return true;
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool HcalDbASCIIIO::getObject(std::istream& fInput, HcalDcsValues * fObject) {
  if (!fObject) return false; //fObject = new HcalDcsValues;
  std::string buffer;
  while (getline(fInput, buffer)) {
    if (buffer.at(0) == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (buffer);
    if (items.size()==0) continue; // blank line

    if (items.size() < 9) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 9 items: subDet, side_ring, slice, type, subChannel, LS, Value, UpperLimit, LowerLimit" << std::endl;
      continue;
    }

    HcalOtherSubdetector subd;
    int sidering;
    unsigned int slice, subchan;
    switch (items[0].at(1)) {
    case 'B':
      subd = HcalDcsBarrel;
      break;
    case 'E':
      subd = HcalDcsEndcap;
      break;
    case 'F':
      subd = HcalDcsForward;
      break;
    case 'O':
      subd = HcalDcsOuter;
      break;
    default:
      continue;
    }
    //from_string<int>(subd, items[0], std::dec);
    from_string<int>(sidering, items[1], std::dec);
    from_string<unsigned int>(slice, items[2], std::dec);
    //from_string<int>(ty, items[3], std::dec);
    from_string<unsigned int>(subchan, items[4], std::dec);

    HcalDcsDetId newId(subd, sidering, slice, 
		       HcalDcsDetId::DcsTypeFromString(items[3]), subchan);

    int LS;
    float val,upper,lower;
    from_string<int>(LS, items[5], std::dec);
    from_string<float>(val, items[6], std::dec);
    from_string<float>(upper, items[7], std::dec);
    from_string<float>(lower, items[8], std::dec);

    HcalDcsValue newVal(newId.rawId(),LS,val,upper,lower);
//     std::cout << buffer << '\n';
//     std::cout << subd << ' ' << sidering << ' ' << slice << ' '
// 	      << ty << ' ' << subchan << ' ' << LS << ' '
// 	      << val << ' ' << upper << ' ' << lower << '\n';
//     std::cout << newId ;
    if (!(fObject->addValue(newVal))) {
      edm::LogWarning("Data Error") << "Data from line " << buffer 
				    << "\nwas not added to the HcalDcsValues object." << std::endl;
    }
//     std::cout << std::endl;
  }
  fObject->sortAll();
  return true;
}

bool HcalDbASCIIIO::dumpObject(std::ostream& fOutput, 
			       HcalDcsValues const& fObject) {
  fOutput << "# subDet side_ring slice type subChan LS Value UpperLimit LowerLimit DcsId\n";
  for(int subd = HcalDcsValues::HcalHB; 
      subd <= HcalDcsValues::HcalHF; ++subd) {
//     std::cout << "subd: " << subd << '\n';
    HcalDcsValues::DcsSet const & vals= 
      fObject.getAllSubdetValues(HcalDcsValues::DcsSubDet(subd));
    for (HcalDcsValues::DcsSet::const_iterator val = vals.begin(); 
	 val != vals.end(); ++val) {
      HcalDcsDetId valId(val->DcsId());

      switch (valId.subdet()) {
      case HcalDcsBarrel:
	fOutput << "HB ";
	break;
      case HcalDcsEndcap:
	fOutput << "HE ";
	break;
      case HcalDcsOuter:
	fOutput << "HO ";
	break;
      case HcalDcsForward:
	fOutput << "HF ";
	break;
      default :
	fOutput << valId.subdet() << ' ';
      }

      if (valId.subdet() == HcalDcsOuter)
	fOutput << valId.ring() << ' ';
      else
	fOutput << valId.zside() << ' ';

      fOutput << valId.slice() << ' ' 
	      << valId.typeString(valId.type()) << ' '
	      << valId.subchannel() << ' ';
      fOutput << val->LS() << ' ' 
	      << val->getValue() << ' '
	      << val->getUpperLimit() << ' '
	      << val->getLowerLimit() << ' ';
      fOutput << std::hex << val->DcsId() << std::dec << '\n';

//       std::cout << valId << ' '
// 		<< valId.subdet() << ' ' 
// 		<< val->LS() << ' ' << val->getValue() << ' '
// 		<< val->getUpperLimit() << ' ' << val->getLowerLimit()
// 		<< std::endl;
    }
  }

  return true;
}


// Format of the ASCII file:
// line# Ring Slice Subchannel Subdetector Eta Phi Depth
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalDcsMap* fObject) {
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size () < 8) {
      if (items.size()==0) continue; // no warning here
      else {
	edm::LogError("MapFormat") << "HcalDcsMap-> Bad line: " << buffer 
				   << "\n line must contain 8 items: line side_ring slice subchannel subdet ieta iphi depth";
	continue;
      }
    }
    //    std::cout << "HcalDcsMap-> processing line: " << buffer << std::endl;
    //int ring = atoi (items [1].c_str());
    int ring = atoi(items[1].c_str());
    unsigned int slice = atoi (items [2].c_str());
    unsigned int subchannel = atoi (items [3].c_str());
    HcalDcsDetId::DcsType type = HcalDcsDetId::DCSUNKNOWN;
//    if(items[4].find("HV")!=std::string::npos){
//      type = HcalDcsDetId::HV;
//    }
//    else if (items[4].find("BV")!=std::string::npos){
//      type = HcalDcsDetId::BV;
//    }
//    else if (items[4].find("CATH")!=std::string::npos){
//      type = HcalDcsDetId::CATH;
//    }
//    else if (items[4].find("DYN7")!=std::string::npos){
//      type = HcalDcsDetId::DYN7;
//    }
//    else if (items[4].find("DYN8")!=std::string::npos){
//      type = HcalDcsDetId::DYN8;
//    }
//    else if (items[4].find("RM_TEMP")!=std::string::npos){
//      type = HcalDcsDetId::RM_TEMP;
//    }
//    else if (items[4].find("CCM_TEMP")!=std::string::npos){
//      type = HcalDcsDetId::CCM_TEMP;
//    }
//    else if (items[4].find("CALIB_TEMP")!=std::string::npos){
//      type = HcalDcsDetId::CALIB_TEMP;
//    }
//    else if (items[4].find("LVTTM_TEMP")!=std::string::npos){
//      type = HcalDcsDetId::LVTTM_TEMP;
//    }
//    else if (items[4].find("TEMP")!=std::string::npos){
//      type = HcalDcsDetId::TEMP;
//    }
//    else if (items[4].find("QPLL_LOCK")!=std::string::npos){
//      type = HcalDcsDetId::QPLL_LOCK;
//    }
//    else if (items[4].find("STATUS")!=std::string::npos){
//      type = HcalDcsDetId::STATUS;
//    }
//    else if (items[4].find("DCS_MAX")!=std::string::npos){
//      type = HcalDcsDetId::DCS_MAX;
//    }
//    else{
//      edm::LogError("MapFormat") << "HcalDcsMap-> Unknown DCS Type, line is not accepted: " << items[4];
//      continue;
//    }
    HcalOtherSubdetector subdet = HcalOtherEmpty;
    if (items[4].find("CALIB")!=std::string::npos){
      subdet = HcalCalibration;
    }
    else if (items[4].find("HB")!=std::string::npos){
      subdet = HcalDcsBarrel;
    }
    else if (items[4].find("HE")!=std::string::npos){
      subdet = HcalDcsEndcap;
    }
    else if (items[4].find("HO")!=std::string::npos){
      subdet = HcalDcsOuter;
    }
    else if (items[4].find("HF")!=std::string::npos){
      subdet = HcalDcsForward;
    }
    else{
      edm::LogError("MapFormat") << "HcalDcsMap-> Unknown subdetector, line is not accepted: " << items[5];
      continue;
    }
    HcalDcsDetId dcsId(subdet, ring, slice, type, subchannel);
    HcalText2DetIdConverter converter (items [4], items [5], items [6], items [7]);
    HcalDetId id(0);
    if (converter.isHcalDetId()){
      id = converter.getId();
    }
    else{
      edm::LogWarning("Invalid HCAL channel") << "HcalDcsMap-> invalid channel: " 
					      << items [4] << '/' 
					      << items [5] << '/'
					      << items [6] << '/' 
					      << items [7] << std::endl; 
      continue;
    }
    fObject->mapGeomId2DcsId(id, dcsId);
  }
  fObject->sort ();
  return true;
}

// Format of the ASCII file:
// line# Ring Slice Subchannel Subdetector Eta Phi Depth
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalDcsMap& fObject) {
  char buf [1024];
  sprintf (buf, "# %7s %10s %6s %8s %7s %5s %5s %6s",
	   "i", "side_ring", "slice", "subchan", "subdet", "ieta", "iphi", "depth");
  fOutput << buf << std::endl;
  HcalDcsMap::const_iterator _line;
  unsigned int line_counter = 0;
  for (_line = fObject.beginById();
       _line != fObject.endById();
       ++_line) {
    HcalDcsDetId dcsId = _line.getHcalDcsDetId();
    //std::string _dcs_type = "DCSUNKNOWN";
    HcalText2DetIdConverter _converter(_line.getHcalDetId());
    sprintf (buf, " %8X %10d %6d %8d %7s %5s %5s %6s",
	     line_counter,
	     dcsId.ring(), // contains zside() already
	     dcsId.slice(),
	     dcsId.subchannel(),
	     _converter.getFlavor().c_str(),
	     _converter.getField1().c_str(),
	     _converter.getField2().c_str(),
	     _converter.getField3().c_str()
	     );
    fOutput << buf << std::endl;
    ++line_counter;
  }
  return true;
}


bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalFlagHFDigiTimeParams* fObject)
{
  
  if (!fObject) return false; //fObject = new HcalFlagHFDigiTimeParams();
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size()==0) continue; // blank line
    if (items.size () != 9) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain at 9 least items: eta, phi, depth, subdet, firstSample, samplesToAdd, ExpectedPeak, MinEnergy, and a set of comma-separated coefficients" << std::endl;
      continue;
    }
    // expects (ieta, iphi, depth, subdet) as first four arguments
    DetId id = HcalDbASCIIIO::getId (items);
    std::vector<double> coef= splitStringToDoubleByComma(items[8].c_str());

    HcalFlagHFDigiTimeParam* fCondObject = new HcalFlagHFDigiTimeParam(id, 
								       atoi (items [4].c_str()), //firstSample
								       atoi (items [5].c_str()), //samplesToAdd
								       atoi (items [6].c_str()), //expectedPeak
								       atof (items [7].c_str()), // minEThreshold
								       coef // coefficients
								  );
    fObject->addValues(*fCondObject);
    delete fCondObject;
  }
  return true;
} // getObject (HcalFlagHFDigiTime)

bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalFlagHFDigiTimeParams& fObject)
{
  char buffer [1024];
  sprintf (buffer, "# %15s %15s %15s %15s  %15s %15s %15s %15s %30s\n", "eta", "phi", "dep", "det", "FirstSample", "SamplesToAdd", "ExpectedPeak","MinEnergy","Coefficients");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    // Dump out channel (ieta,iphi,depth,subdet) info
    HcalDbASCIIIO::dumpId (fOutput, *channel);
    // Dump out values for channel
    sprintf (buffer, " %15u %15u %15u %15f",
	     fObject.getValues (*channel)->HFdigiflagFirstSample(), 
	     fObject.getValues (*channel)->HFdigiflagSamplesToAdd(), 
	     fObject.getValues (*channel)->HFdigiflagExpectedPeak(), 
	     fObject.getValues (*channel)->HFdigiflagMinEThreshold() 
	     );

    fOutput<<buffer; // dump flag reco values to buffer
    fOutput<<"               "; // simple spacer

    std::vector<double> coef=fObject.getValues(*channel)->HFdigiflagCoefficients();
    for (std::vector<double>::size_type x=0;x<coef.size();++x)
      {
	// dump comma-separated list of coefficients
	fOutput<<coef[x];
	if (x<coef.size()-1) // add commas where necessary
	  fOutput<<",";
      }
    sprintf(buffer,"\n");
    fOutput << buffer;
  }
  return true;
}


