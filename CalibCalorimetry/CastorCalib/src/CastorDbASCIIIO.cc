//
// F.Ratnikov (UMd), Oct 28, 2005
//
#include <vector>
#include <string>
#include <cstdio>

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/CastorElectronicsId.h"
#include "CalibFormats/CastorObjects/interface/CastorText2DetIdConverter.h"

#include "CondFormats/CastorObjects/interface/AllObjects.h"
#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace CastorDbASCIIIO {
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
  class CastorElectronicsIdLess {
  public:
    bool operator () (CastorElectronicsId first, CastorElectronicsId second) const {
      return
	first.readoutVMECrateId () != second.readoutVMECrateId () ? first.readoutVMECrateId () < second.readoutVMECrateId () :
	first.htrSlot () != second.htrSlot () ? first.htrSlot () < second.htrSlot () :
	first.htrTopBottom () != second.htrTopBottom () ? first.htrTopBottom () < second.htrTopBottom () :
	first.fiberIndex () != second.fiberIndex () ? first.fiberIndex () < second.fiberIndex () :
	first.fiberChanId () < second.fiberChanId ();
    }
  };

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

DetId getId (const std::vector <std::string> & items) {
  CastorText2DetIdConverter converter (items [3], items [0], items [1], items [2]);
  return converter.getId ();
}

void dumpId (std::ostream& fOutput, DetId id) {
  CastorText2DetIdConverter converter (id);
  char buffer [1024];
  sprintf (buffer, "  %15s %15s %15s %15s",
	   converter.getField1 ().c_str (), converter.getField2 ().c_str (), converter.getField3 ().c_str (),converter.getFlavor ().c_str ());  
  fOutput << buffer;
}

template <class S,class T> 
bool getCastorObject (std::istream& fInput, T* fObject) {
  if (!fObject) fObject = new T;
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.empty()) continue; // blank line
    if (items.size () < 8) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 8 items: eta, phi, depth, subdet, 4x values" << std::endl;
      continue;
    }
    DetId id = getId (items);
    
//    if (fObject->exists(id) )
//      edm::LogWarning("Redefining Channel") << "line: " << buffer << "\n attempts to redefine data. Ignored" << std::endl;
//    else
//      {
    S fCondObject(id, atof (items [4].c_str()), atof (items [5].c_str()), atof (items [6].c_str()), atof (items [7].c_str()));
    fObject->addValues(fCondObject);
	//      }
  }

  return true;
}

template <class T>
bool dumpCastorObject (std::ostream& fOutput, const T& fObject) {
  char buffer [1024];
  sprintf (buffer, "# %15s %15s %15s %15s %8s %8s %8s %8s %10s\n", "eta", "phi", "dep", "det", "cap0", "cap1", "cap2", "cap3", "DetId");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  //std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       ++channel) {
    const float* values = fObject.getValues (*channel)->getValues ();
    if (values) {
      dumpId (fOutput, *channel);
      sprintf (buffer, " %8.5f %8.5f %8.5f %8.5f %10X\n",
	       values[0], values[1], values[2], values[3], channel->rawId ());
      fOutput << buffer;
    }
  }
  return true;
}

template <class S,class T> 
bool getCastorSingleFloatObject (std::istream& fInput, T* fObject) {
  if (!fObject) fObject = new T;
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.empty()) continue; // blank line
    if (items.size () < 5) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 5 items: eta, phi, depth, subdet, value" << std::endl;
      continue;
    }
    DetId id = getId (items);
    
//    if (fObject->exists(id) )
//      edm::LogWarning("Redefining Channel") << "line: " << buffer << "\n attempts to redefine data. Ignored" << std::endl;
//    else
//      {
    S fCondObject(id, atof (items [4].c_str()) );
    fObject->addValues(fCondObject);
    //      }
  }
  return true;
}

template <class T>
bool dumpCastorSingleFloatObject (std::ostream& fOutput, const T& fObject) {
  char buffer [1024];
  sprintf (buffer, "# %15s %15s %15s %15s %8s %10s\n", "eta", "phi", "dep", "det", "value", "DetId");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       ++channel) {
    const float value = fObject.getValues (*channel)->getValue ();
    dumpId (fOutput, *channel);
    sprintf (buffer, " %8.5f %10X\n",
	     value, channel->rawId ());
    fOutput << buffer;
  }
  return true;
}

template <class S,class T> 
bool getCastorSingleIntObject (std::istream& fInput, T* fObject, S* fCondObject) {
  if (!fObject) fObject = new T;
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.empty()) continue; // blank line
    if (items.size () < 5) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 5 items: eta, phi, depth, subdet, value" << std::endl;
      continue;
    }
    DetId id = getId (items);
    
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
bool dumpCastorSingleIntObject (std::ostream& fOutput, const T& fObject) {
  char buffer [1024];
  sprintf (buffer, "# %15s %15s %15s %15s %8s %10s\n", "eta", "phi", "dep", "det", "value", "DetId");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       ++channel) {
    const int value = fObject.getValues (*channel)->getValue ();
    dumpId (fOutput, *channel);
    sprintf (buffer, " %15d %10X\n",
	     value, channel->rawId ());
    fOutput << buffer;
  }
  return true;
}


bool getObject (std::istream& fInput, CastorGains* fObject) {return getCastorObject<CastorGain> (fInput, fObject);}
bool dumpObject (std::ostream& fOutput, const CastorGains& fObject) {return dumpCastorObject (fOutput, fObject);}
bool getObject (std::istream& fInput, CastorGainWidths* fObject) {return getCastorObject<CastorGainWidth> (fInput, fObject);}
bool dumpObject (std::ostream& fOutput, const CastorGainWidths& fObject) {return dumpCastorObject (fOutput, fObject);}

bool getObject (std::istream& fInput, CastorSaturationCorrs* fObject) {return getCastorSingleFloatObject<CastorSaturationCorr> (fInput, fObject);}
bool dumpObject (std::ostream& fOutput, const CastorSaturationCorrs& fObject) {return dumpCastorSingleFloatObject (fOutput, fObject);}



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool getObject (std::istream& fInput, CastorPedestals* fObject) {
  if (!fObject) fObject = new CastorPedestals(false);
  char buffer [1024];

  while (fInput.getline(buffer, 1024)) {
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.empty()) continue; // blank line
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
    if (items.empty()) continue; // blank line
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
	CastorPedestal* fCondObject = new CastorPedestal(id, atof (items [4].c_str()), atof (items [5].c_str()), 
						     atof (items [6].c_str()), atof (items [7].c_str()), 
						     0., 0., 0., 0. );
	fObject->addValues(*fCondObject);
	delete fCondObject;
      }
    else // new format with widths
      {
	CastorPedestal* fCondObject = new CastorPedestal(id, atof (items [4].c_str()), atof (items [5].c_str()), 
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


bool dumpObject (std::ostream& fOutput, const CastorPedestals& fObject) {
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
       ++channel) {
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
bool getObject (std::istream& fInput, CastorChannelQuality* fObject) 
{
  if (!fObject) fObject = new CastorChannelQuality;
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.empty()) continue; // blank line
    if (items.size () < 5) {
      edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 5 items: eta, phi, depth, subdet, GOOD/BAD/HOT/DEAD" << std::endl;
      continue;
    }
    DetId id = getId (items);
    
    if (fObject->exists(id) ) {
      edm::LogWarning("Redefining Channel") << "line: " << buffer << "\n attempts to redefine data. Ignored" << std::endl;
      continue;
    }
//    else
//      {
    uint32_t mystatus;
    CastorChannelStatus* fCondObject = nullptr;
    if (items[4].substr(0,2)=="0x") {
       sscanf(items[4].c_str(),"%X", &mystatus);
       fCondObject = new CastorChannelStatus(id,mystatus);
    }
    else if (isalpha(items[4].c_str()[0])) {
       fCondObject = new CastorChannelStatus(id, items[4]);
    }
    else {
       sscanf(items[4].c_str(),"%u", &mystatus);
       fCondObject = new CastorChannelStatus(id,mystatus);
    }
    fObject->addValues(*fCondObject);
    delete fCondObject;
	//      }
  }
  return true;
}


bool dumpObject (std::ostream& fOutput, const CastorChannelQuality& fObject) {
  char buffer [1024];
  sprintf (buffer, "# %15s %15s %15s %15s %15s %10s\n", "eta", "phi", "dep", "det", "value", "DetId");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       ++channel) {
    const int value = fObject.getValues (*channel)->getValue ();
    dumpId (fOutput, *channel);
    sprintf (buffer, " %15X %10X\n",
	     value, channel->rawId ());
    fOutput << buffer;
  }
  return true;
}


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool getObject (std::istream& fInput, CastorPedestalWidths* fObject) {
  if (!fObject) fObject = new CastorPedestalWidths(false);
  char buffer [1024];
  int linecounter = 0;

  while (fInput.getline(buffer, 1024)) {
    linecounter++;
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.empty()) continue; // blank line
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
    if (items.empty()) continue; // blank line
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
	CastorPedestalWidth values(id);
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
	CastorPedestalWidth values(id);
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

bool dumpObject (std::ostream& fOutput, const CastorPedestalWidths& fObject) {
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
       ++channel) {
    const CastorPedestalWidth* item = fObject.getValues (*channel);
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
bool getObject (std::istream& fInput, CastorQIEData* fObject) {
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.empty()) continue;
    if (items [0] == "SHAPE") { // basic shape
      if (items.size () < 33) {
	edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 33 items: SHAPE  32 x low QIE edges for first 32 bins" << std::endl;
	continue;
      }
      //float lowEdges [32];
      //int i = 32;
      //while (--i >= 0) lowEdges [i] = atof (items [i+1].c_str ());
      //      fObject->setShape (lowEdges);
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
	CastorQIECoder coder (id.rawId ());
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
	fObject->addCoder (coder);
//      }
    }
  }
  fObject->sort ();
  return true;
}

bool dumpObject (std::ostream& fOutput, const CastorQIEData& fObject) {
  char buffer [1024];
  fOutput << "# QIE basic shape: SHAPE 32 x low edge values for first 32 channels" << std::endl;
  sprintf (buffer, "SHAPE ");
  fOutput << buffer;
  for (unsigned bin = 0; bin < 32; bin++) {
    sprintf (buffer, " %8.5f", fObject.getShape ().lowEdge (bin));
    fOutput << buffer;
  }
  fOutput << std::endl;

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
       ++channel) {
    const CastorQIECoder* coder = fObject.getCoder (*channel);
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
    fOutput << std::endl;
  }
  return true;
}

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool getObject (std::istream& fInput, CastorCalibrationQIEData* fObject) {
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
      CastorCalibrationQIECoder coder (id.rawId ());
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

bool dumpObject (std::ostream& fOutput, const CastorCalibrationQIEData& fObject) {
  char buffer [1024];
  fOutput << "# QIE data in calibration mode" << std::endl;
  sprintf (buffer, "# %15s %15s %15s %15s %288s\n", 
	   "eta", "phi", "dep", "det", "32 x charges");
  fOutput << buffer;
  std::vector<DetId> channels = fObject.getAllChannels ();
  std::sort (channels.begin(), channels.end(), DetIdLess ());
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       ++channel) {
    const CastorCalibrationQIECoder* coder = fObject.getCoder (*channel);
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
bool getObject (std::istream& fInput, CastorElectronicsMap* fObject) {
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size () < 12) {
      if (items.empty()) continue; // no warning here
      if (items.size()<9) {
	edm::LogError("MapFormat") << "CastorElectronicsMap-> line too short: " << buffer;
	continue;
      }
      if (items[8]=="NA" || items[8]=="NT") {
	while (items.size()<12) items.push_back(""); // don't worry here
      } else if (items[8]=="HT") {
	if (items.size()==11) items.push_back("");
	else {
	  edm::LogError("MapFormat") << "CastorElectronicsMap-> Bad line: " << buffer 
				     << "\n HT line must contain at least 11 items: i  cr sl tb dcc spigot fiber fiberchan subdet=HT ieta iphi";
	  continue;
	}
      } else {
	edm::LogError("MapFormat") << "CastorElectronicsMap-> Bad line: " << buffer 
				   << "\n line must contain 12 items: i  cr sl tb dcc spigot fiber fiberchan subdet ieta iphi depth";
	continue;
      }
    }
    //    std::cout << "CastorElectronicsMap-> processing line: " << buffer << std::endl;
    int crate = atoi (items [1].c_str());
    int slot = atoi (items [2].c_str());
    int top = 1;
    if (items [3] == "b") top = 0;
    int dcc = atoi (items [4].c_str());
    int spigot = atoi (items [5].c_str());
    CastorElectronicsId elId;
    if (items[8] == "HT" || items[8] == "NT") {
      int slb = atoi (items [6].c_str());
      int slbCh = atoi (items [7].c_str());
      elId=CastorElectronicsId(slbCh, slb, spigot, dcc,crate,slot,top);
    } else {
      int fiber = atoi (items [6].c_str());
      int fiberCh = atoi (items [7].c_str());

      elId=CastorElectronicsId(fiberCh, fiber, spigot, dcc);
      elId.setHTR (crate, slot, top);
    }

    // first, handle undefined cases
    if (items [8] == "NA") { // undefined channel
      fObject->mapEId2chId (elId, DetId (HcalDetId::Undefined));
    } else if (items [8] == "NT") { // undefined trigger channel
      fObject->mapEId2tId (elId, DetId (HcalTrigTowerDetId::Undefined));
    } else {
      CastorText2DetIdConverter converter (items [8], items [9], items [10], items [11]);
      if (converter.isHcalCastorDetId ()) { 
	fObject->mapEId2chId (elId, converter.getId ());
      }
      else {
	edm::LogWarning("Format Error") << "CastorElectronicsMap-> Unknown subdetector: " 
		  << items [8] << '/' << items [9] << '/' << items [10] << '/' << items [11] << std::endl; 
      }
    }
  }
  fObject->sort ();
  return true;
}

bool dumpObject (std::ostream& fOutput, const CastorElectronicsMap& fObject) {
  std::vector<CastorElectronicsId> eids = fObject.allElectronicsId ();
  char buf [1024];
  // changes by Jared, 6.03.09/(included 25.03.09)
  //  sprintf (buf, "#%10s %6s %6s %6s %6s %6s %6s %6s %15s %15s %15s %15s",
  sprintf (buf, "# %7s %3s %3s %3s %4s %7s %10s %14s %7s %5s %5s %6s",
	   "i", "cr", "sl", "tb", "dcc", "spigot", "fiber/slb", "fibcha/slbcha", "subdet", "ieta", "iphi", "depth");
  fOutput << buf << std::endl;

  for (unsigned i = 0; i < eids.size (); i++) {
    CastorElectronicsId eid = eids[i];
    if (eid.isTriggerChainId()) {
      DetId trigger = fObject.lookupTrigger (eid);
      if (trigger.rawId ()) {
	CastorText2DetIdConverter converter (trigger);
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
	CastorText2DetIdConverter converter (channel);
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

bool getObject (std::istream& fInput, CastorRecoParams* fObject) {
	if (!fObject) fObject = new CastorRecoParams();
	char buffer [1024];
	while (fInput.getline(buffer, 1024)) {
		if (buffer [0] == '#') continue; //ignore comment
		std::vector <std::string> items = splitString (std::string (buffer));
		if (items.empty()) continue; // blank line
		if (items.size () < 6) {
			edm::LogWarning("Format Error") << "Bad line: " << buffer << "\n line must contain 6 items: eta, phi, depth, subdet, firstSample, samplesToAdd" << std::endl;
		    	continue;
		}
		DetId id = getId (items);
	      
		CastorRecoParam* fCondObject = new CastorRecoParam(id, atoi (items [4].c_str()), atoi (items [5].c_str()) );
		fObject->addValues(*fCondObject);
		delete fCondObject;
	}
	return true;
}

bool dumpObject (std::ostream& fOutput, const CastorRecoParams& fObject) {
	char buffer [1024];
	sprintf (buffer, "# %15s %15s %15s %15s %18s %15s %10s\n", "eta", "phi", "dep", "det", "firstSample", "samplesToAdd", "DetId");
	fOutput << buffer;
	std::vector<DetId> channels = fObject.getAllChannels ();
	std::sort (channels.begin(), channels.end(), DetIdLess ());
	for (std::vector<DetId>::iterator channel = channels.begin();channel != channels.end();++channel) {
		dumpId (fOutput, *channel);
		sprintf (buffer, " %15d %15d %16X\n",
		fObject.getValues (*channel)->firstSample(), fObject.getValues (*channel)->samplesToAdd(), channel->rawId ());
		fOutput << buffer;
	}
	return true;
}

}
