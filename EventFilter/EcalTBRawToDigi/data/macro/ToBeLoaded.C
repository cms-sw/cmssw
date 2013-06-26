#include<string>
//#include<vector>
#include <sstream>
std::string IntToString(int num)
 {
   std::ostringstream myStream; //creates an ostringstream object
   myStream << num << std::flush;

  /*
   * outputs the number into the string stream and then flushes
   * the buffer (makes sure the output is put into the stream)
   */

   return(myStream.str()); //returns the string form of the stringstream object
 }

// 
// File H4Geom.cxx
//
/*! \class H4Geom
 * \brief A helper class with geometry information of the super module
 *
 * $Date: 2010/10/21 17:33:47 $
 * $Author: wmtan $
 *
 * Crystal numbering schemes during automn 2004 (SM10):
 * - <b>Conventions</b>
 *   SM Crystal Number range is [1, 1700]
 *   SM Tower Number range is [1, 68]
 *   eta range is [0, 84]
 *   phi range is [0, 19]
 * - <b>Crystal Number in Readout-Order</b>
 *   Crystal numbering in readout order is very similar to 2003 with two
 *   different orders according to the LVRB position (right or left) on the
 *   trigger tower.
 *   \verbatim
 *                  <- eta
 *   +--+--+--+--+--+      +--+--+--+--+--+
 *   | 4| 5|14|15|24|      |20|19|10| 9| 0|
 *   +--+--+--+--+--+      +--+--+--+--+--+
 *   | 3| 6|13|16|23|      |21|18|11| 8| 1|
 *   +--+--+--+--+--+      +--+--+--+--+--+  ||
 *   | 2| 7|12|17|22|      |22|17|12| 7| 2|  || phi
 *   +--+--+--+--+--+      +--+--+--+--+--+  \/
 *   | 1| 8|11|18|21|      |23|16|12| 6| 3| 
 *   +--+--+--+--+--+      +--+--+--+--+--+
 *   | 0| 9|10|19|20|      |24|15|14| 5| 4| 
 *   +--+--+--+--+--+      +--+--+--+--+--+
 *      left tower            right tower
 *   \endverbatim
 *   In order to know if a trigger tower (from 1 -> 68) is right or left, the
 *   following logic can be applied:
 *     if ((smTower - 1)/4 < 3 || ((smTower - 1)/4 - 3)%4 >= 2)         
 *       left tower
 *     else
 *       right tower
 * - <b>Super Module Crystal Number</b> 
 *   Organization of crystals in the lines of a supermodule has been changed
 *   in 2004:
 *   \verbatim
 *(i = 84, j = 0)          <- eta               (i = 0, j = 0)
 *      +----+----+     +----+----+----+----+----+----+
 *      |1681|1661| ... | 101|  81|  61|  41|  21|   1|
 *      +----+----+     +----+----+----+----+----+----+
 *      |1682|1662| ... | 102|  82|  62|  42|  22|   2|
 *      +----+----+     +----+----+----+----+----+----+  
 *      ...                                              ||      |
 *      +----+----+     +----+----+----+----+----+----+  || phi  |
 *      |1698|1678| ... | 118|  98|  78|  58|  38|  18|  \/      |t
 *      +----+----+     +----+----+----+----+----+----+          |o
 *      |1699|1679| ... | 119|  99|  79|  59|  39|  19|          |w
 *      +----+----+     +----+----+----+----+----+----+          |e
 *      |1700|1680| ... | 120| 100|  80|  60|  40|  20|          |r
 *      +----+----+     +----+----+----+----+----+----+          |
 *(i = 84, j = 19)                              (i = 0, j = 19)  
 *                           --------- tower ----------
 *   \endverbatim
 *   smCrystalNumber is obtained from i and j:
 *     smCrystalNumber = i*5*4 + j + 1
 *   conversly:
 *     i = (smCrystalNumber - 1)/(5*4)
 *     j = (smCrystalNumber - 1)%(5*4)
 *
 *   the logic to determine i and j from smTower and crystal readout number is:
 *     if (tower is left)
 *       i = (smTower - 1)/4 * 5 + (24 - crystal)/5
 *     else
 *       i = (smTower - 1)/4 * 5 + crystal/5
 *     if (tower is left && (crystal/5)%2 = 0 || 
 *         tower is right && (crystal/5)%2 = 1)
 *       j = (smTower - 1)%4 * 5 + 4 - crystal%5
 *     else
 *       j = (smTower - 1)%4 * 5 + crystal%5
 *   conversly
 *     smTower = i/5 * 4 + j/5 + 1
 *     if (tower is left)
 *       if ((i%5)%2 = 0)
 *         crystal = (4 - i%5)*5 + 4 - j%5
 *       else
 *         crystal = (4 - i%5)*5 + j%5
 *     else
 *       if ((i%5)%2 = 0)
 *         crystal = i%5 * 5 + j%5
 *       else
 *         crystal = i%5 * 5 + 4 - j%5 
 *
 *
 * Crystal numbering schemes in 2003:
 * - <b>Crystal Number in Readout-Order</b>
 *   The data on the RRF files contain the crystal information in
 *   readout order. The physical position of the crystals in readout
 *   order depends then on the VFE-Module and Tower-Number. There
 *   exist two different kinds of towers, one where the low voltage
 *   board sits on the left side, and one where the board sits on the
 *   right side. The following scheme shows the position in eta and
 *   phi of the crystals given their readout crystal number: 
 *   \verbatim
 *                  <- eta
 *   +--+--+--+--+--+      +--+--+--+--+--+
 *   | 4| 5|14|15|24|      |20|19|10| 9| 0|
 *   +--+--+--+--+--+      +--+--+--+--+--+
 *   | 3| 6|13|16|23|      |21|18|11| 8| 1|
 *   +--+--+--+--+--+      +--+--+--+--+--+  ^
 *   | 2| 7|12|17|22|      |22|17|12| 7| 2|  | phi
 *   +--+--+--+--+--+      +--+--+--+--+--+  |
 *   | 1| 8|11|18|21|      |23|16|12| 6| 3| 
 *   +--+--+--+--+--+      +--+--+--+--+--+
 *   | 0| 9|10|19|20|      |24|15|14| 5| 4| 
 *   +--+--+--+--+--+      +--+--+--+--+--+
 *      left tower            right tower
 *   \endverbatim
 *   <b>If in this documentation a crystal number is mentioned, then by
 *   default, the readout number is meant. </b>
 * - <b>Super Module Crystal Number</b> 
 *   Organization of crystals in the lines of a supermodule:
 *   \verbatim
 *                 <- eta
 *      +----+----+     +----+----+----+----+----+----+
 *      |1699|1698| ... |1620|1919|1618|1617|1616|1615|
 *      +----+----+     +----+----+----+----+----+----+
 *      |1614|1613| ... |1535|1534|1533|1532|1531|1530|
 *      +----+----+     +----+----+----+----+----+----+  ^
 *      ...                                              | phi  |
 *      +----+----+     +----+----+----+----+----+----+  |      |
 *      | 254| 253| ... | 175| 174| 173| 172| 171| 170|         |t
 *      +----+----+     +----+----+----+----+----+----+         |o
 *      | 169| 168| ... |  90|  89|  88|  87|  86|  85|         |w
 *      +----+----+     +----+----+----+----+----+----+         |e
 *      |  84|  83| ... |   5|   4|   3|   2|   1|   0|         |r
 *      +----+----+     +----+----+----+----+----+----+         |
 *
 *                           --------- tower ----------
 *   \endverbatim
 *   The above numbering scheme schows the numbers of a crystal in the
 *   super module (SM). The sm-crystal-number allows to uniquely
 *   identify the crystal in the H4 test-beams since only one SM is put
 *   into the beam at a time. The sm-crystal-number allows navigation
 *   across the boarders of the towers and is used on the ERF level by
 *   void H4RecEnergy::getCrystal(H4RecXtal * oneXtal, int smCrystal).
 * - <b>Crystal Number in Geometrical Order</b>
 *   The geometrical crystal number within a tower reflects the
 *   physical position of a crystal and is therefore used on the ERF
 *   level where higher level information is of interest. The
 *   geometrical ordering counts crystals as:
 *   \verbatim
 *        <- eta
 *   +--+--+--+--+--+
 *   |24|19|14| 9| 4|
 *   +--+--+--+--+--+
 *   |23|18|13| 8| 3|  ^
 *   +--+--+--+--+--+  | phi
 *   |22|17|12| 7| 2|  |
 *   +--+--+--+--+--+ 
 *   |21|16|11| 6| 1| 
 *   +--+--+--+--+--+ 
 *   |20|15|10| 5| 0| 
 *   +--+--+--+--+--+ 
 *   \endverbatim
 *   This numbering scheme is offered by 
 *   void H4RecEnergy::getTower(int towerNb, H4RecXtal * tower[])
 *
 * The H4Geom class is providing methods to convert these numbers into
 * each other.
*/

#include "./Geom.h"
#include <iostream>

using namespace std;

const int H4Geom::crystalChannelMap[5][5] = {
  { 4, 5, 14, 15, 24},
  { 3, 6, 13, 16, 23},
  { 2, 7, 12, 17, 22},
  { 1, 8, 11, 18, 21},
  { 0, 9, 10, 19, 20}
};

const int H4Geom::crystalMap[25] = {
  340, 255, 170,  85,   0,
    1,  86, 171, 256, 341,
  342, 257, 172,  87,   2,
    3,  88, 173, 258, 343,
  344, 259, 174,  89,   4
};
const int H4Geom::WhichHalf[69] = {
  0, // TT 0 does not exist 
  1,1,1,1,
  2,2,1,1,
  2,2,1,1,
  2,2,1,1,
  2,2,1,1,
  2,2,1,1,
  2,2,1,1,
  2,2,1,1,
  2,2,1,1,
  2,2,1,1,
  2,2,1,1,
  2,2,1,1,
  2,2,1,1,
  2,2,1,1,
  2,2,1,1,
  2,2,1,1,
  2,2,1,1};


H4Geom::GeomPeriod_t H4Geom::geometry_ = H4Geom::Undef;

// initializes geometry since we always use Automn2004
// (previously done in init() )
H4Geom::H4Geom()
{ 
  GeomPeriod_t ForEcalMonitoring;    ForEcalMonitoring = Automn2004;
  H4Geom::SetGeomPeriod(ForEcalMonitoring);
}

//! does nothing
H4Geom::~H4Geom()
{ }

//! Initialize geometry with config file
bool H4Geom::init()
{
  //   if (gConfigParser->isDefined("Geometry::Period"))
  //     geometry_ = H4Geom::GeomPeriod_t(gConfigParser->
  // 				     readIntOption("Geometry::Period"));
  //   else
  //     return false;
  
  // initialization of period moved to constructor, since we always use Automn2004
  //   GeomPeriod_t ForEcalMonitoring;    ForEcalMonitoring = Automn2004;
  //   H4Geom::SetGeomPeriod(ForEcalMonitoring);
  return true;
}

//! Retuns the crystal number in the super module for a given
//! tower number in the super module and crystal number in the tower
int H4Geom::getSMCrystalNumber(int tower, int crystal) const 
{
  if (!IsGeomPeriodDefined()) 
    {cout << "[H4Geom] geometry period not defined, aborting.\t (should be  ForEcalMonitoring = Automn2004;)" << endl;
      abort();}

  if (geometry_ == H4Geom::Spring2004) {
    int smCrystalNbr = 0;
    if ((crystal/5)%2 == 0)
      smCrystalNbr = kCrystalsPerTower - 
	int(crystal/kCardsPerTower)*kChannelsPerCard - 
	crystal%kChannelsPerCard + (tower - 1)*kCrystalsPerTower;
    else
      smCrystalNbr = kCrystalsPerTower + 1 - 
	(int(crystal/kCardsPerTower) + 1)*kChannelsPerCard + 
	crystal%kChannelsPerCard + (tower - 1)*kCrystalsPerTower;
    return smCrystalNbr;
  } else if (geometry_ == H4Geom::Automn2004) {
    int eta = (tower - 1)/kTowersInPhi*kCardsPerTower;
    int phi = (tower - 1)%kTowersInPhi*kChannelsPerCard;
    if (rightTower(tower))
      eta += crystal/kCardsPerTower;
    else
      eta += (kCrystalsPerTower - 1 - crystal)/kCardsPerTower;
    if (rightTower(tower) && (crystal/kCardsPerTower)%2 == 1 ||
	!rightTower(tower) && (crystal/kCardsPerTower)%2 == 0)
      phi += (kChannelsPerCard - 1 - crystal%kChannelsPerCard);
    else
      phi += crystal%kChannelsPerCard;
    return eta*kChannelsPerCard*kTowersInPhi + phi + 1;
  } else {
    int towerId = tower - 1;
    int line = towerId%4;
    int column = towerId/4;
    int lowerRight = 5*line*85 + column*5;
    if (rightTower(tower))
      return lowerRight+crystalMap[crystal];
    else
      return lowerRight+crystalMap[24-crystal];
  }
}


//! Retuns the crystal number in the super module for a given
//! tower number in the super module and crystal number in the tower
int H4Geom::getSMCrystalNumber(int tower, int strip_id, int crystal_id) const {
  if (
      !(0< strip_id && strip_id <6
	&& 0< crystal_id && crystal_id <6)
      )    {
    cout << "[H4geom] invalid strip_id or crystal_id; ";
    cout << "strip= "<<strip_id << " crystal_id= " << crystal_id << endl ;
    //    abort();
    return -1;
  }
  else
    {
      //      cout << "[H4Geom][getSMCrystalNumber] doing nothing for now" << endl;
      //       strip_id and crystal_id --> {1,2,3,4,5}
      //       crystal_id = 5 - crystal_id + 1;
      //       int  crystalInTower = 5*(strip_id-1) + crystal_id-1 ;
      int  crystalInTower = 5*(strip_id-1) + crystal_id -1;
      //      cout << "[H4geom][debug] crystalInTower is: " << crystalInTower << endl;
      return 
	H4Geom::getSMCrystalNumber(tower, crystalInTower); // should be ok for TT right
    }
}


//! Retuns the crystal number in the super module for a given
//! tower number in the super module and crystal number in the tower
void H4Geom::getTowerStripChannelNumber(int& tower, int& strip_id, int& crystal_id, int sm_num) const {
  int CryNum;
  getTowerCrystalNumber(tower,CryNum,sm_num);
  strip_id   = CryNum/5 + 1;
  crystal_id = CryNum%5 + 1;
}




//! Returns the crystal number in a tower for a given
//! crystal number in the super module
void H4Geom::getTowerCrystalNumber(int &tower, int &crystal, int smCrystal) const 
{
  if (!IsGeomPeriodDefined()) 
    {cout << "[H4Geom] geometry period not defined, aborting.\t (should be  ForEcalMonitoring = Automn2004;)" << endl;
      abort();}
  if (geometry_ == H4Geom::Spring2004) {
    // We do not know where the supercrystal is...
    tower = (smCrystal - 1)/kCrystalsPerTower + 1; 
    smCrystal -= (tower - 1)*kCrystalsPerTower;
    if (((smCrystal - 1)/kCardsPerTower)%2 == 0)
      crystal = (kCardsPerTower - 
		 int((smCrystal - 1)/kCardsPerTower))*kCardsPerTower - 
                (smCrystal - 1)%kCardsPerTower - 1;
    else
      crystal = (kCardsPerTower - 1 - 
		 int((smCrystal - 1)/kCardsPerTower))*kCardsPerTower + 
                (smCrystal - 1)%kCardsPerTower;
    return;
  } else if (geometry_ == H4Geom::Automn2004) {
    int eta = (smCrystal - 1)/(kChannelsPerCard*kTowersInPhi);
    int phi = (smCrystal - 1)%(kChannelsPerCard*kTowersInPhi);
    tower = eta/kCardsPerTower*kTowersInPhi + phi/kChannelsPerCard + 1;
    if (rightTower(tower)) {
      crystal = (eta%kCardsPerTower)*kChannelsPerCard;
      if ((eta%kChannelsPerCard)%2 == 0)
	crystal += phi%kChannelsPerCard;
      else
	crystal += kChannelsPerCard - 1 - phi%kChannelsPerCard;
    } else {
      crystal = (kCardsPerTower - 1 - eta%kCardsPerTower)*kChannelsPerCard;
      if ((eta%kChannelsPerCard)%2 == 0)
	crystal += kChannelsPerCard - 1 - phi%kChannelsPerCard;
      else
	crystal += phi%kChannelsPerCard;
    }
  } else {
    int line = smCrystal/85;
    int cInLine = smCrystal%85;
    
    tower= 1 + (cInLine/5) * 4 + line/5; // tower number in SM
    crystal = crystalChannelMap[line%5][cInLine%5];
    if (leftTower(tower)) crystal = 24-crystal ;
  }
  return;
}


//! Returns the crystal number (readout order) in a tower 
//! for a given position in the tower (crystalNbGeom=0 is the 
//! lower-right corner and crystalNbGeom=24 is the upper-left corner,
//! see scheme in H4Geom::getTower)
int H4Geom::getTowerCrystalNumber(int smTowerNb, int crystalNbGeom) const
{
  if (crystalNbGeom < 0 || crystalNbGeom >= kCrystalsPerTower) return -1 ;
  int column = crystalNbGeom/kCardsPerTower;
  int line = crystalNbGeom%kChannelsPerCard;
  int crystal = crystalChannelMap[line][column];
  if (leftTower(smTowerNb)) crystal = kCrystalsPerTower - 1 - crystal;
  return crystal;
}


//! Returns the crystal coordinates (eta, phi index) for a given
//! crystal number in the super module
void H4Geom::getCrystalCoord(int &eta, int &phi, int smCrystal) const
{
  if (geometry_ == H4Geom::Spring2004) {
    eta = (smCrystal - 1)/kChannelsPerCard; // arbitrary units
    phi = kChannelsPerCard - 1 - 
      ((smCrystal - 1)%kCrystalsPerTower)%kChannelsPerCard;
  } else if (geometry_ == H4Geom::Automn2004) {
    eta = (smCrystal - 1)/(kChannelsPerCard*kTowersInPhi);
    phi = (smCrystal - 1)%(kChannelsPerCard*kTowersInPhi);
  } else {
    eta = smCrystal%kCrystalsInEta;
    phi = smCrystal/kCrystalsInEta;
  }
}


//! Retuns the crystal number in the super module for given coordinates
int H4Geom::getSMCrystalFromCoord(int eta, int phi) const
{
  if (geometry_ == H4Geom::Spring2004) {
    if (eta >= 9 || eta < 0 || phi >= 4 || phi < 0) 
      return -1; // Only two supercrystals
    return eta*kCardsPerTower + (kChannelsPerCard - phi);
  }
  if (eta < 0 || eta >= kCrystalsInEta || phi < 0 || phi >= kCrystalsInPhi) 
    return -1; // non real smCrystal
  else {
    if (geometry_ == H4Geom::Automn2004)
      return eta*kChannelsPerCard*kTowersInPhi + phi + 1;
    else
      return eta + kCrystalsInEta*phi;
  }
}


//! Returns left neighbour of a sm crystal.
//! Input and output are crystal numbers in the super module.
//! A negative output means outside of the supermodule.
int H4Geom::getLeft(int smCrystal) const
{
  if (!IsGeomPeriodDefined()) 
    {cout << "[H4Geom] geometry period not defined, aborting.\t (should be  ForEcalMonitoring = Automn2004;)" << endl;
      abort();}
  if (geometry_ == H4Geom::Spring2004) {
    smCrystal += kChannelsPerCard;
    if (smCrystal > 2*kCrystalsPerTower || smCrystal < 1) return -1;
    return smCrystal;
  } else if (geometry_ == H4Geom::Automn2004) {
    smCrystal += kCrystalsInPhi;
    if (smCrystal > kCrystals) return -1;
    return smCrystal;
  } else {
    smCrystal++;
    if (!(smCrystal % kCrystalsInEta)) return -1;
    if (smCrystal >= kCrystals || smCrystal < 0) return -1;
    return smCrystal;
  }
}


//! Returns right neighbour of a sm crystal.
//! Input and output are crystal numbers in the super module.
//! A negative output means outside of the supermodule.
int H4Geom::getRight(int smCrystal) const
{
  if (!IsGeomPeriodDefined()) 
    {cout << "[H4Geom] geometry period not defined, aborting.\t (should be  ForEcalMonitoring = Automn2004;)" << endl;
      abort();}
  if (geometry_ == H4Geom::Spring2004) {
    smCrystal -= kChannelsPerCard;
    if (smCrystal > 2*kCrystalsPerTower || smCrystal < 1) return -1;
    return smCrystal;
  } else if (geometry_ == H4Geom::Automn2004) {
    smCrystal -= kCrystalsInPhi;
    if (smCrystal < 0) return -1;
    return smCrystal;
  } else {
    smCrystal--;
    if (smCrystal % kCrystalsInEta == kCrystalsInEta - 1) return -1;
    if (smCrystal >= kCrystals || smCrystal < 0) return -1;
    return smCrystal;
  }
}


//! Returns upper neighbour of a sm crystal.
//! Input and output are crystal numbers in the super module.
//! A negative output means outside of the supermodule.
int H4Geom::getUpper(int smCrystal) const{
  if (!IsGeomPeriodDefined()) 
    {cout << "[H4Geom] geometry period not defined, aborting.\t (should be  ForEcalMonitoring = Automn2004;)" << endl;
      abort();}
  if (geometry_ == H4Geom::Spring2004) {
    smCrystal--;
    if (smCrystal > 2*kCrystalsPerTower || smCrystal < 1) return -1;
    return smCrystal;
  } else if (geometry_ == H4Geom::Automn2004) {
    if ((smCrystal - 1)%kCrystalsInPhi == 0) return -1; // phi = 0
    smCrystal--;
    return smCrystal;
  } else {
    smCrystal += kCrystalsInEta;
    if (smCrystal >= kCrystals || smCrystal < 0) return -1;
    return smCrystal;
  }
}


//! Returns lower neighbour of a sm crystal.
//! Input and output are crystal numbers in the super module.
//! A negative output means outside of the supermodule.
int H4Geom::getLower(int smCrystal) const
{
  if (!IsGeomPeriodDefined()) 
    {cout << "[H4Geom] geometry period not defined, aborting.\t (should be  ForEcalMonitoring = Automn2004;)" << endl;
      abort();}
  if (geometry_ == H4Geom::Spring2004) {
    smCrystal ++;
    if (smCrystal > 2*kCrystalsPerTower || smCrystal < 1) return -1;
    return smCrystal;
  } else if (geometry_ == H4Geom::Automn2004) {
    if (smCrystal%kCrystalsInPhi == 0) return -1; // phi = 19
    smCrystal++;
    return smCrystal;
  } else {
    smCrystal -= kCrystalsInEta;
    if (smCrystal >= kCrystals || smCrystal < 0) return -1;
    return smCrystal;
  }
}


//! Returns left neighbour of a crystal referenced by its coordinates.
//! New coordonates overwrite the old ones. No check is done to see
//! if it corresponds to a real crystal. To be used with caution. 
void H4Geom::mvLeft(int &eta, int &phi) const
{
  if (eta >= kCrystalsInEta)
    std::cout << "H4Geom::mvLeft: eta is too large " << eta << std::endl;
  eta++;
}

//! Returns right neighbour of a crystal referenced by its coordinates.
//! New coordonates overwrite the old ones. No check is done to see
//! if it corresponds to a real crystal. To be used with caution. 
void H4Geom::mvRight(int &eta, int &phi) const
{
  if (eta < 0)
    std::cout << "H4Geom::mvRight: eta is too small " << eta << std::endl;
  eta--;
}

//! Returns upper neighbour of a crystal referenced by its coordinates.
//! New coordonates overwrite the old ones. No check is done to see
//! if it corresponds to a real crystal. To be used with caution. 
void H4Geom::mvUp(int &eta, int &phi) const
{
  if (phi >= kCrystalsInPhi)
    std::cout << "H4Geom::mvUp: phi is too large " << phi << std::endl;
  phi++;
}


//! Returns lower neighbour of a crystal referenced by its coordinates.
//! New coordonates overwrite the old ones. No check is done to see
//! if it corresponds to a real crystal. To be used with caution. 
void H4Geom::mvDown(int &eta, int &phi) const
{
  if (phi < 0)
    std::cout << "H4Geom::mvDown: phi is too small " << phi << std::endl;
  phi--;
}


//! Returns the 25 crystals of tower towerNb in the super module.
//! Output are crystal numbers in the super module.
//! By default, the order in the output array (tower) corresponds to
//! geometric order (index 0 is lower-right corner).
//! if order=readout, the order in the output array (tower) 
//! corresponds to the readout scheme (depends on the kind of tower)
/**
 * the geometric order is defined by:
 * \verbatim
 *  +--+--+--+--+--+
 *  |24|19|14| 9| 4|
 *  +--+--+--+--+--+
 *  |23|18|13| 8| 3|
 *  +--+--+--+--+--+  
 *  |22|17|12| 7| 2| 
 *  +--+--+--+--+--+ 
 *  |21|16|11| 6| 1| 
 *  +--+--+--+--+--+ 
 *  |20|15|10| 5| 0| 
 *  +--+--+--+--+--+ 
 * \endverbatim
 */ 
void H4Geom::getTower(int * tower, int towerNb, std::string order) const 
{
  if (!IsGeomPeriodDefined()) 
    {cout << "[H4Geom] geometry period not defined, aborting.\t (should be  ForEcalMonitoring = Automn2004;)" << endl;
      abort();}
  if (order == "readout") {
    for (int crystalNb = 0; crystalNb < kCrystalsPerTower; crystalNb++)
      tower[crystalNb] = getSMCrystalNumber(towerNb, crystalNb);
  } else {
    int towerId = towerNb - 1;
    int line = towerId%kTowersInPhi;
    int column = towerId/kTowersInPhi;
    if (geometry_ == H4Geom::Automn2004) {
      int smLowerRight = column*kCardsPerTower*kCrystalsInPhi + 
	                 (line + 1)*kChannelsPerCard;
      for (int i = 0; i < kCrystalsPerTower; i++)
	tower[i] = smLowerRight - i%kChannelsPerCard + 
                   i/kChannelsPerCard*kCrystalsInPhi;
    } else {
      int smLowerRight = kChannelsPerCard*line*kCrystalsInEta + 
	                 column*kCardsPerTower;
      for (int i = 0; i < kCrystalsPerTower; i++)
	tower[i] = i%kChannelsPerCard * kCrystalsInEta + 
	           i/kChannelsPerCard + smLowerRight;
    }
  }
  return;
}


//! Returns the 5 crystals belonging to the same VFE board as smCrystal.
//! Input and output are crystal numbers in the super module.
//! By default, the order in the output array (VFE) corresponds to
//! The geometric order (index 0 is lower-right corner).
//! if order=readout, the order in the output array (VFE) 
//! corresponds to the readout scheme (depends on the kind of tower)
/**
 * the geometric order is defined by:
 * \verbatim
 *  +-+
 *  |4|
 *  +-+
 *  |3|
 *  +-+  
 *  |2| 
 *  +-+ 
 *  |1| 
 *  +-+ 
 *  |0| 
 *  +-+ 
 * \endverbatim
 */ 
void H4Geom::getVFE(int * VFE, int smCrystal, std::string order) const 
{
  int towerNb, crystalNb;
  getTowerCrystalNumber(towerNb, crystalNb, smCrystal);
  int VFEnb = crystalNb/kChannelsPerCard;
  if (order == "readout") {
    for (crystalNb = 0; crystalNb < kChannelsPerCard; crystalNb++) 
      VFE[crystalNb] = getSMCrystalNumber(towerNb, 
					  kChannelsPerCard*VFEnb+crystalNb);
  } else {
    int eta, phi;
    getCrystalCoord(eta, phi, smCrystal);
    int smLower = eta+85*5*(phi/5);	
    for (int i = 0; i < 5; i++)
      VFE[i] =  i*85 + smLower;
  }  
}

//! Returns sm crystal numbers for crystals in a window of
//! size widthxheight centered around a given smCrystal.
//! width and height must be odd.
//! The order in the output array (window) is defined 
//! by the geometric order (index 0 is lower-right corner).
/**
 * the geometric order is defined by:
 * \verbatim
 *         width (w)
 * <--------------------->
 * +---------------------+   -
 * | h*w . . . . .  .   h|   |
 * | .              .   .|   |
 * | .              .   .|   | h
 * | .              .   .|   | e
 * | . . . . x . .  .   .|   | i (h)
 * | .              .   .|   | g
 * | .              .   .|   | h
 * | .              .   .|   | t
 * | .              .   1|   | 
 * | . . . . . . . h+1  0|   |
 * +---------------------+   -
 *
 *    x: central crystal     
 * \endverbatim
 */
void H4Geom::getWindow(int * window, int smCrystal, int width, int height) const
{
  if (width <= 0 || width%2 == 0) {
    std::cout << "H4Geom::getWindow, width should be >0 and odd!" << std::endl;
    return;
  }
  if (height <= 0 || height%2 == 0) {
    std::cout << "H4Geom::getWindow, height should be >0 and odd!" 
	      << std::endl;
    return;
  }
  int eta, phi;
  getCrystalCoord(eta, phi, smCrystal);
  // get lower-right corner
  int eta0 = eta - (width - 1)/2;
  int phi0 = phi - (height - 1)/2;
  int index = 0;
  eta = eta0;
  phi = phi0;
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      window[index] = getSMCrystalFromCoord(eta, phi);
      phi++;
      index++;
    }
    eta++;
    phi = phi0;
  }
}


//! Tests if low voltage board is on the right size of the tower.
//! Readout scheme depends on that.
/**
 * the readout order for a "right-tower" is defined by:
 * \verbatim
 *  +--+--+--+--+--+
 *  |20|19|10| 9| 0|
 *  +--+--+--+--+--+
 *  |21|18|11| 8| 1|
 *  +--+--+--+--+--+  
 *  |22|17|12| 7| 2| 
 *  +--+--+--+--+--+ 
 *  |23|16|12| 6| 3| 
 *  +--+--+--+--+--+ 
 *  |24|15|14| 5| 4| 
 *  +--+--+--+--+--+ 
 * \endverbatim
 */ 
bool H4Geom::rightTower(int tower) const 
{
  if (!IsGeomPeriodDefined()) 
    {cout << "[H4Geom] geometry period not defined, aborting.\t (should be  ForEcalMonitoring = Automn2004;)" << endl;
      abort();}
  if (geometry_ == H4Geom::Automn2004) {
    if ((tower - 1)/kTowersInPhi < 3 ||
	((tower - 1)/kTowersInPhi - 3)%4 >= 2)
      return false;
    else
      return true;
  }
  if ((tower>12 && tower<21) || (tower>28 && tower<37) ||
      (tower>44 && tower<53) || (tower>60 && tower<69))
    return true;
  else
    return false;
}

//! Tests if low voltage board is on the left size of the tower.
//! Readout scheme depends on that.
/**
 * the readout order for a "left-tower" is defined by:
 * \verbatim
 *  +--+--+--+--+--+
 *  | 4| 5|14|15|24|
 *  +--+--+--+--+--+
 *  | 3| 6|13|16|23|
 *  +--+--+--+--+--+  
 *  | 2| 7|12|17|22| 
 *  +--+--+--+--+--+ 
 *  | 1| 8|11|18|21| 
 *  +--+--+--+--+--+ 
 *  | 0| 9|10|19|20| 
 *  +--+--+--+--+--+ 
 * \endverbatim
 */ 
bool H4Geom::leftTower(int tower) const
{
  return !rightTower(tower);
}

void H4Geom::SetGeomPeriod(GeomPeriod_t geometry)
{
  geometry_ = geometry;
}

bool H4Geom::IsGeomPeriodDefined() const {
  if (geometry_ == H4Geom::Undef) {
    std::cout << "Class H4Geom : geometry is not defined. You should call in"
	      << " your program H4Geom::init (and fill the [Geometry] section"
	      << " of your config file) or call H4Geom::SetGeomPeriod." 
	      << std::endl;
    return false;
  }
  return true;
}

int H4Geom::getHalf (int TT){
  if(TT < 1 || TT >68){cout <<"TT num out of range: "<<TT<<endl; return 0;}
  return WhichHalf[TT];
}
