#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"

#include <sstream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

namespace {
  //const bool phase1 = false;
  const bool pilot_blade = true;  
}

PixelEndcapName::PixelEndcapName(const DetId & id, const TrackerTopology* tt, bool phase)
  : PixelModuleName(false), phase1(phase)
{

  //PXFDetId cmssw_numbering(id);
  int side     = tt->pxfSide(id);
  int tmpBlade = tt->pxfBlade(id);
  theDisk      = tt->pxfDisk(id);
  thePlaquette = tt->pxfModule(id);
  bool outer = false;   // outer means with respect to the LHC ring (x - axis)


  if(phase1) {  // phase 1

    // this still has to be modified so blades start from 1 on the top
    if (tmpBlade>=7 && tmpBlade<=17) {
      outer = true;
      theBlade = tmpBlade-6;                      //7...17-->1...11
    } else if (tmpBlade>=32 && tmpBlade<=48) {
      outer = true;
      theBlade = 60-tmpBlade;                     //32...48-->28...12
    } else if( tmpBlade<=6 ) {
      theBlade = 7-tmpBlade;                      //1...6-->6...1
    } else if( tmpBlade>=18 && tmpBlade<=31 ) {
      theBlade = 38-tmpBlade;                     //18...31-->20...7
    } else if( tmpBlade>=49 && tmpBlade<=56 ) {
      theBlade = 77-tmpBlade;                     //49...56-->28...21
    } //  iasonas2-end

    thePannel = tt->pxfPanel(id); // this is the ring
    //thePannel = tt->pxfRing(id); // this is the ring

  } else { // phase0

    // hack for the pilot blade
    if(pilot_blade && theDisk==3 ) { // do only for disk 3
      //cout<<tmpBlade<<" "<<theDisk<<endl;
      if(tmpBlade>=1 && tmpBlade<=4) {
	// convert the sequential counting of pilot blades to std. cmssw convention
	if(tmpBlade<3) tmpBlade +=3;
	else  tmpBlade +=13;
      } else {
	edm::LogError ("Bad PilotBlade blade number ") 
	  << tmpBlade;      
      }
    }

    if (tmpBlade >= 7 && tmpBlade <= 18) {
      outer = true;
      theBlade = tmpBlade-6;
    } else if( tmpBlade <=6 ) { 
      theBlade = 7-tmpBlade; 
    } else if( tmpBlade >= 19) { 
      theBlade = 31-tmpBlade; 
    } 

    thePannel    = tt->pxfPanel(id);
    
  } // end phase1

       if( side == 1 &&  outer ) thePart = mO;
  else if( side == 1 && !outer ) thePart = mI;
  else if( side == 2 &&  outer ) thePart = pO;
  else if( side == 2 && !outer ) thePart = pI;

}

PixelEndcapName::PixelEndcapName(const DetId & id, bool phase)
  : PixelModuleName(false), phase1(phase)
{
  PXFDetId cmssw_numbering(id);
  int side = cmssw_numbering.side();
  int tmpBlade = cmssw_numbering.blade();
  thePlaquette = cmssw_numbering.module();
  theDisk = cmssw_numbering.disk();
  bool outer = false;   // outer means with respect to the LHC ring (x - axis)

  if(phase1) { // phase1
    // this still has to be modified so blades start from 1 on the top
    if (tmpBlade>=7 && tmpBlade<=17) {
      outer = true;   // outer means with respect to the LHC ring (x - axis)
      theBlade = tmpBlade-6;                      //7...17-->1...11
    } else if (tmpBlade>=32 && tmpBlade<=48) {
      outer = true;
      theBlade = 60-tmpBlade;                     //32...48-->28...12
    } else if( tmpBlade<=6 ) {
      theBlade = 7-tmpBlade;                      //1...6-->6...1
    } else if( tmpBlade>=18 && tmpBlade<=31 ) {
      theBlade = 38-tmpBlade;                     //18...31-->20...7
    } else if( tmpBlade>=49 && tmpBlade<=56 ) {
      theBlade = 77-tmpBlade;                     //49...56-->28...21
    } //  iasonas2-end
    
    thePannel = cmssw_numbering.panel();  // this is really the ring

  } else { // phase 0

    // hack for the pilot blade
    if(pilot_blade && theDisk==3 ) { // do only for disk 3
      if(tmpBlade>=1 && tmpBlade<=4) {
	// convert the sequential counting of pilot blades to std. cmssw convention
	if(tmpBlade<3) tmpBlade +=3;
	else  tmpBlade +=13;
      } else {
	edm::LogError ("Bad PilotBlade blade number ") 
	  << tmpBlade;      
      }
    }

    if (tmpBlade >= 7 && tmpBlade <= 18) {
      outer = true;   // outer means with respect to the LHC ring (x - axis)
      theBlade = tmpBlade-6;
    } else if( tmpBlade <=6 ) { 
      theBlade = 7-tmpBlade; 
    } else if( tmpBlade >= 19) { 
      theBlade = 31-tmpBlade; 
    }

    thePannel = cmssw_numbering.panel();
 
  } // end phase1

  if( side == 1 &&  outer ) thePart = mO;
  else if( side == 1 && !outer ) thePart = mI;
  else if( side == 2 &&  outer ) thePart = pO;
  else if( side == 2 && !outer ) thePart = pI;
 
}

// constructor from name string
PixelEndcapName::PixelEndcapName(std::string name)
  : PixelModuleName(false), PixelEndcapNameBase(mO, 0, 0, 0, 0) {

  // parse the name string
  // first, check to make sure this is an FPix name, should start with "FPix_"
  // also check to make sure the needed parts are present
  if ( (name.substr(0, 5) != "FPix_") ||
       (name.find("_B") == string::npos) || 
       (name.find("_D") == string::npos) ||
       (name.find("_BLD") == string::npos) || 
       (name.find("_PNL") == string::npos) ||
       ( (phase1  && name.find("_RNG") == string::npos) ) ||
       ( (!phase1 && name.find("_PLQ") == string::npos) ) ) {
    edm::LogError ("BadNameString|SiPixel") 
      << "Bad name string in PixelEndcapName::PixelEndcapName(std::string): "
      << name;
    return;
  }

  // strip off ROC part if it's there
  if (name.find("_ROC") != string::npos)
    name = name.substr(0, name.find("_ROC"));

  // get the half cylinder
  string hcString = name.substr(name.find("_B")+2, name.find("_D")-name.find("_B")-2);
  if (hcString == "mO") thePart = mO;
  else if (hcString == "mI") thePart = mI;
  else if (hcString == "pO") thePart = pO;
  else if (hcString == "pI") thePart = pI;
  else {
    edm::LogError ("BadNameString|SiPixel") 
      << "Unable to determine half cylinder in PixelEndcapName::PixelEndcapName(std::string): "
      << name;
  }

  // get the disk
  string diskString = name.substr(name.find("_D")+2, name.find("_BLD")-name.find("_D")-2);
  if (diskString == "1") theDisk = 1;
  else if (diskString == "2") theDisk = 2;
  else if (diskString == "3") theDisk = 3;
  else {
    edm::LogError ("BadNameString|SiPixel") 
      << "Unable to determine disk number in PixelEndcapName::PixelEndcapName(std::string): "
      << name;
  }

  // get the blade
  string bladeString = name.substr(name.find("_BLD")+4, name.find("_PNL")-name.find("_BLD")-4);
  // since atoi() doesn't report errors, do it the long way
  if (bladeString == "1") theBlade = 1;
  else if (bladeString == "2") theBlade = 2;
  else if (bladeString == "3") theBlade = 3;
  else if (bladeString == "4") theBlade = 4;
  else if (bladeString == "5") theBlade = 5;
  else if (bladeString == "6") theBlade = 6;
  else if (bladeString == "7") theBlade = 7;
  else if (bladeString == "8") theBlade = 8;
  else if (bladeString == "9") theBlade = 9;
  else if (bladeString == "10") theBlade = 10;
  else if (bladeString == "11") theBlade = 11;
  else if (bladeString == "12") theBlade = 12;
  else if (bladeString == "13") theBlade = 13;
  else if (bladeString == "14") theBlade = 14;
  else if (bladeString == "15") theBlade = 15;
  else if (bladeString == "16") theBlade = 16;
  else if (bladeString == "17") theBlade = 17;
  else {
    edm::LogError ("BadNameString|SiPixel") 
      << "Unable to determine blade number in PixelEndcapName::PixelEndcapName(std::string): "
      << name;
  }


  if(phase1) { // phase1
    string ringString = name.substr(name.find("_RNG")+4, name.size()-name.find("_RNG")-4);
    if (ringString == "1")      thePannel = 1; // code ring in the pannel
    else if (ringString == "2") thePannel = 2;
    else {
      edm::LogError ("BadNameString|SiPixel") 
	<< "Unable to determine ring number in PixelEndcapName::PixelEndcapName(std::string): "
	<< name;
    }

  } else { // phase 0

    // find the panel
    string panelString = name.substr(name.find("_PNL")+4, name.find("_PLQ")-name.find("_PNL")-4);
    if (panelString == "1") thePannel = 1;
    else if (panelString == "2") thePannel = 2;
    else {
      edm::LogError ("BadNameString|SiPixel") 
	<< "Unable to determine panel number in PixelEndcapName::PixelEndcapName(std::string): "
	<< name;
    }
  } // end phase1

  // find the plaquette
  string plaquetteString = name.substr(name.find("_PLQ")+4, name.size()-name.find("_PLQ")-4);
  if (plaquetteString == "1") thePlaquette = 1;
  else if (plaquetteString == "2") thePlaquette = 2;
  else if (plaquetteString == "3") thePlaquette = 3;
  else if (plaquetteString == "4") thePlaquette = 4;
  else {
    edm::LogError ("BadNameString|SiPixel") 
      << "Unable to determine plaquette number in PixelEndcapName::PixelEndcapName(std::string): "
      << name;
  }


} // PixelEndcapName::PixelEndcapName(std::string name)

PixelModuleName::ModuleType  PixelEndcapName::moduleType() const
{
  ModuleType type = v1x2;

  if(phase1) { // phase1

    type = v2x8;

  } else { // phase 0

    if (pannelName() == 1) {
      if (plaquetteName() == 1)      { type = v1x2; }
      else if (plaquetteName() == 2) { type = v2x3; }
      else if (plaquetteName() == 3) { type = v2x4; }
      else if (plaquetteName() == 4) { type = v1x5; }
    } else {
      if (plaquetteName() == 1)      { type = v2x3; }
      else if (plaquetteName() == 2) { type = v2x4; }
      else if (plaquetteName() == 3) { type = v2x5; }
    }

    // hack for the pilot blade
    if(pilot_blade && theDisk==3 ) {type=v2x8;} // do only for disk 3

  } // end phase1

  return type;
}

bool PixelEndcapName::operator== (const PixelModuleName & o) const
{
  if (!o.isBarrel()) {
    const PixelEndcapName * other = dynamic_cast<const PixelEndcapName *>(&o);
    return (    other 
             && thePart      == other->thePart
             && theDisk      == other->theDisk
             && theBlade     == other->theBlade
             && thePannel    == other->thePannel
             && thePlaquette == other->thePlaquette ); 
  } else return false;
}


string PixelEndcapName::name() const 
{
  std::ostringstream stm;

  stm <<"FPix_B"<<thePart<<"_D"<<theDisk<<"_BLD"<<theBlade<<"_PNL"<<thePannel<<"_PLQ"<<thePlaquette;
  return stm.str();
}

// return the DetId
DetId PixelEndcapName::getDetId(const TrackerTopology* tt) {
  
  uint32_t side = 0;
  uint32_t disk = 0;
  uint32_t blade = 0;
  uint32_t panel = 0;
  uint32_t module = 0;

  // figure out the side
  HalfCylinder hc = halfCylinder();
  if (hc == mO || hc == mI) side = 1;
  else if (hc == pO || hc == pI) side = 2;
  
  // get disk/blade/panel/module numbers from PixelEndcapName object
  disk = static_cast<uint32_t>(diskName());
  uint32_t tmpBlade = static_cast<uint32_t>(bladeName());
  panel = static_cast<uint32_t>(pannelName());

  // convert blade numbering to cmssw convention
  bool outer = false;
  outer = (hc == mO) || (hc == pO);

  if(phase1) { // phase1

    // this still has to be modified so blades start from 1 on the top
    if (outer) {
      if          (tmpBlade>=7 && tmpBlade<=17)    theBlade = tmpBlade+6;
      else if     (tmpBlade>=32 && tmpBlade<=48)   theBlade = 60-tmpBlade;
    } else { //inner
      if          (tmpBlade<=6 )                   theBlade = 7-tmpBlade;
      else if     (tmpBlade>=18 && tmpBlade<=31)   theBlade = 38-tmpBlade;
      else if     (tmpBlade>=49 && tmpBlade<=56)   theBlade = 77-tmpBlade;
    }

    module = static_cast<uint32_t>(pannelName());
  
  } else { // phase 0
 
    if (outer) {
      blade = tmpBlade + 6;
    } else { // inner
      if (tmpBlade <= 6) blade = 7 - tmpBlade;
      else if (tmpBlade <= 12) blade = 31 - tmpBlade;
    }

    // hack for the pilot blade
    if(pilot_blade && theDisk==3 ) { // do only for disk 3
      //cout<<tmpBlade<<" "<<blade<<endl;
      if(blade<=5) blade -=3;
      else         blade -=13;
      //cout<<tmpBlade<<" "<<blade<<endl;
    }

    module = static_cast<uint32_t>(plaquetteName());
  } // end phase1


  // create and return the DetId
  DetId id = tt->pxfDetId(side, disk, blade, panel, module);

  return id;

} // PXFDetId PixelEndcapName::getDetId()


// return the DetId
PXFDetId PixelEndcapName::getDetId() {
  
  uint32_t side = 0;
  uint32_t disk = 0;
  uint32_t blade = 0;
  uint32_t panel = 0;
  uint32_t module = 0;

  // figure out the side
  HalfCylinder hc = halfCylinder();
  if (hc == mO || hc == mI) side = 1;
  else if (hc == pO || hc == pI) side = 2;
  
  // get disk/blade/panel/module numbers from PixelEndcapName object
  disk = static_cast<uint32_t>(diskName());
  uint32_t tmpBlade = static_cast<uint32_t>(bladeName());
  panel = static_cast<uint32_t>(pannelName());

  // convert blade numbering to cmssw convention
  bool outer = false;
  outer = (hc == mO) || (hc == pO);

  if(phase1) { // phase1

    // this still has to be modified so blades start from 1 on the top
    if (outer) {
      if          (tmpBlade>=7 && tmpBlade<=17)    theBlade = tmpBlade+6;
      else if     (tmpBlade>=32 && tmpBlade<=48)   theBlade = 60-tmpBlade;
    } else { //inner
      if          (tmpBlade<=6 )                   theBlade = 7-tmpBlade;
      else if     (tmpBlade>=18 && tmpBlade<=31)   theBlade = 38-tmpBlade;
      else if     (tmpBlade>=49 && tmpBlade<=56)   theBlade = 77-tmpBlade;
    }

    module = static_cast<uint32_t>(pannelName());
  
  } else { // phase 0
    if (outer) {
      blade = tmpBlade + 6;
    } else { // inner
      if (tmpBlade <= 6) blade = 7 - tmpBlade;
      else if (tmpBlade <= 12) blade = 31 - tmpBlade;
    }

    // hack for the pilot blade
    if(pilot_blade && theDisk==3 ) { // do only for disk 3
      //cout<<tmpBlade<<" "<<blade<<endl;
      if(blade<=5) blade -=3;
      else         blade -=13;
      //cout<<tmpBlade<<" "<<blade<<endl;
    }

    module = static_cast<uint32_t>(plaquetteName());
  } // end phase1

  // create and return the DetId
  return PXFDetId(side, disk, blade, panel, module);

} // PXFDetId PixelEndcapName::getDetId()
