
//
// F.Ratnikov (UMd), Oct 28, 2005
// $Id: HcalDbProducer.h,v 1.2 2005/10/04 18:03:03 fedor Exp $
//
#include <vector>
#include <string>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDetIdDb.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

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

HcalDetId getId (const std::vector <std::string> & items) {
  int eta = atoi (items [0].c_str());
  int phi = atoi (items [1].c_str());
  int depth = atoi (items [2].c_str());
  HcalSubdetector subdet = HcalBarrel;
  if (items [3] == "HE") subdet = HcalEndcap;
  else if (items [3] == "HF") subdet = HcalForward;
  return HcalDetId (subdet, eta, phi, depth);
}

void dumpId (std::ostream& fOutput, HcalDetId id) {
  char buffer [1024];
  std::string subdet = "HB";
  if (id.subdet() == HcalEndcap) subdet = "HE";
  else if (id.subdet() == HcalForward) subdet = "HF";
  sprintf (buffer, "  %4i %4i %4i %4s",
	   id.ieta(), id.iphi(), id.depth (), subdet.c_str ());
   fOutput << buffer;
}

template <class T> 
bool getHcalObject (std::istream& fInput, T* fObject) {
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size () < 8) {
      std::cerr << "Bad line: " << buffer << "\n line must contain 8 items: eta, phi, depth, subdet, 4x values" << std::endl;
      continue;
    }
    fObject->addValue (HcalDetIdDb::HcalDetIdDb (getId (items)), 
		       atof (items [4].c_str()), atof (items [5].c_str()), 
		       atof (items [6].c_str()), atof (items [7].c_str()));
  }
  fObject->sort ();
  return true;
}

template <class T>
bool dumpHcalObject (std::ostream& fOutput, const T& fObject) {
  char buffer [1024];
  sprintf (buffer, "# %4s %4s %4s %4s %8s %8s %8s %8s %10s\n", "eta", "phi", "dep", "det", "cap1", "cap2", "cap3", "cap4", "HcalDetId");
  fOutput << buffer;
  std::vector<unsigned long> channels = fObject.getAllChannels ();
  for (std::vector<unsigned long>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    HcalDetId id ((uint32_t) *channel);
    const float* values = fObject.getValues (*channel);
    if (values) {
      dumpId (fOutput, id);
      sprintf (buffer, " %8.5f %8.5f %8.5f %8.5f %10X\n",
	       values[0], values[1], values[2], values[3], HcalDetIdDb::HcalDetIdDb (id));
      fOutput << buffer;
    }
  }
  return true;
}


bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalPedestals* fObject) {return getHcalObject (fInput, fObject);}
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalPedestals& fObject) {return dumpHcalObject (fOutput, fObject);}
bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalPedestalWidths* fObject) {return getHcalObject (fInput, fObject);}
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalPedestalWidths& fObject) {return dumpHcalObject (fOutput, fObject);}
bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalGains* fObject) {return getHcalObject (fInput, fObject);}
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalGains& fObject) {return dumpHcalObject (fOutput, fObject);}
bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalGainWidths* fObject) {return getHcalObject (fInput, fObject);}
bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalGainWidths& fObject) {return dumpHcalObject (fOutput, fObject);}

bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalQIEShape* fObject) {
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size () < 33) {
      std::cerr << "Bad line: " << buffer << "\n line must contain 32 items: counts for first 33 QIE channels" << std::endl;
      continue;
    }
    for (unsigned i = 0; i <= 32; i++)  fObject->setLowEdge (atof (items [i].c_str ()), i);
    return true;
  }
  return false;
}

bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalQIEShape& fObject) {
  fOutput << "# QIE Shape: low edges for first 33 channels" << std::endl;
  for (unsigned i = 0; i <= 32; i++)  fOutput << fObject.lowEdge (i) << ' ';
  fOutput << std::endl;
  return true;
}

bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalQIEData* fObject) {
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size () < 36) {
      std::cerr << "Bad line: " << buffer << "\n line must contain 36 items: eta, phi, depth, subdet, 4 capId x 4 Ranges x offsets, 4 capId x 4 Ranges x slopes" << std::endl;
      continue;
    }
    float values [32];
    for (int i = 0; i < 32; i++) values [i] = atof (items [i+4].c_str());
    fObject->addValue (HcalDetIdDb::HcalDetIdDb (getId (items)),
		       &values[0], &values[16]);
  }
  fObject->sort ();
  return true;
}

bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalQIEData& fObject) {
  char buffer [1024];
  sprintf (buffer, "# %4s %4s %4s %4s %36s %36s %36s %36s %36s %36s %36s %36s\n", 
	   "eta", "phi", "dep", "det", 
	   "4 x offsets cap1", "4 x offsets cap2", "4 x offsets cap3", "4 x offsets cap4",
	   "4 x slopes cap1", "4 x slopes cap2", "4 x slopes cap3", "4 x slopes cap4");
  fOutput << buffer;
  std::vector<unsigned long> channels = fObject.getAllChannels ();
  for (std::vector<unsigned long>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    HcalDetId id ((uint32_t) *channel);
    const float* offsets = fObject.getOffsets (*channel);
    const float* slopes = fObject.getSlopes (*channel);
    if (offsets && slopes) {
      dumpId (fOutput, id);
      for (int i = 0; i < 16; i++) sprintf (buffer, " %8.5f", offsets [i]); fOutput << buffer;
      for (int i = 0; i < 16; i++) sprintf (buffer, " %8.5f", slopes [i]); fOutput << buffer;
      sprintf (buffer, " %10X\n", HcalDetIdDb::HcalDetIdDb (id)); fOutput << buffer;
    }
  }
  return true;
}

bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalChannelQuality* fObject) {
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size () < 5) {
      std::cerr << "Bad line: " << buffer << "\n line must contain 5 items: eta, phi, depth, subdet, GOOD/BAD/HOT/DEAD" << std::endl;
      continue;
    }
    HcalChannelQuality::Quality value (HcalChannelQuality::UNKNOWN);
    for (int i = 0; i < (int) HcalChannelQuality::END; i++) {
      if (items [4] == std::string (HcalChannelQuality::str ((HcalChannelQuality::Quality) i))) {
	value = (HcalChannelQuality::Quality) i;
      }
    }
    fObject->setChannel (HcalDetIdDb::HcalDetIdDb (getId (items)),
			 value);
  }
  fObject->sort ();
  return true;
}

bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalChannelQuality& fObject) {
  char buffer [1024];
  sprintf (buffer, "# %4s %4s %4s %4s %8s\n", 
	   "eta", "phi", "dep", "det", 
	   "quality");
  fOutput << buffer;
  std::vector<unsigned long> channels = fObject.getAllChannels ();
  for (std::vector<unsigned long>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    HcalDetId id ((uint32_t) *channel);
    dumpId (fOutput, id);
    sprintf (buffer, " %8s\n", HcalChannelQuality::str (fObject.quality (*channel)));
  }
  return true;
}

bool HcalDbASCIIIO::getObject (std::istream& fInput, HcalElectronicsMap* fObject) {
  char buffer [1024];
  while (fInput.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size () < 12) {
      if (items.size () > 0) {
	std::cerr << "Bad line: " << buffer << "\n line must contain 12 items: i  cr sl tb dcc spigot fiber fiberchan subdet ieta iphi depth" << std::endl;
      }
      continue;
    }
    int crate = atoi (items [1].c_str());
    int slot = atoi (items [2].c_str());
    int top = 1;
    if (items [3] == "b") top = 0;
    int dcc = atoi (items [4].c_str());
    int spigot = atoi (items [5].c_str());
    int fiber = atoi (items [6].c_str());
    int fiberCh = atoi (items [7].c_str());
    HcalSubdetector subdet = HcalBarrel;
    if (items [8] == "HE") subdet = HcalEndcap;
    else if (items [8] == "HF") subdet = HcalForward;
    else if (items [8] == "HT") subdet = HcalTriggerTower;
    int eta = atoi (items [9].c_str());
    int phi = atoi (items [10].c_str());
    int depth = atoi (items [11].c_str());
    
    HcalElectronicsId elId (fiberCh, fiber, spigot, dcc);
    elId.setHTR (crate, slot, top);
    if (subdet == HcalTriggerTower) {
      HcalTrigTowerDetId trigId (eta, phi);
      fObject->mapEId2tId (elId (), trigId.rawId());
    }
    else {
      HcalDetId chId (subdet, eta, phi, depth);
      fObject->mapEId2chId (elId (), chId.rawId());
    }
  }
  return true;
}

bool HcalDbASCIIIO::dumpObject (std::ostream& fOutput, const HcalElectronicsMap& fObject) {
  std::cerr << "HcalDbASCIIIO::dumpObject for HcalElectronicsMap is not implemented" << std::endl;
  return false;
}
