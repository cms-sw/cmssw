/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Detector.h
 *
 *    Description:  CSC detector functions.
 *
 *        Version:  1.0
 *        Created:  05/19/2008 10:52:21 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_Detector_H
#define CSCDQM_Detector_H

#include <math.h>
#include <float.h>
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>

#ifdef CSC_RENDER_PLUGIN
#include "CSCDQM_Utility.h"
#include "CSCDQM_WiregroupData.h"
#else
#include "DQM/CSCMonitorModule/interface/CSCDQM_Utility.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_WiregroupData.h"
#endif

#define PI 3.14159256

namespace cscdqm {

/**
 * Number of Detector Components.
 */

#define N_SIDES    2
#define N_STATIONS 4
#define N_RINGS    3
#define N_CHAMBERS 36
#define N_LAYERS   6
#define N_CFEBS    5
#define N_HVS      5

/** Size of the address (number of components) */
#define ADDR_SIZE  7

/** Number of addressing elements in detector */
#define N_ELEMENTS 9540
//(7740 + 1800)

/**
 * Partition function shortcuts
 */

#define PARTITION_INDEX(x,y)  (x * partitions_y + y)
#define PARTITION_STEP_X      (5.0 / partitions_x)
#define PARTITION_STEP_Y      ((2.0 * 3.14159) / partitions_y)

/**
 * @brief  Mask of the address which is used to switch on and off appropriate Address fields.
 */
struct AddressMask {
  bool side;
  bool station;
  bool ring;
  bool chamber;
  bool layer;
  bool cfeb;
  bool hv;
};

/**
 * @brief  Structure to store detector addresses of any granularity: from
 * whole detector to the single HV element.
 */
struct Address {

  unsigned int side;
  unsigned int station;
  unsigned int ring;
  unsigned int chamber;
  unsigned int layer;
  unsigned int cfeb;
  unsigned int hv;

  AddressMask mask;

  bool operator== (const Address& a) const {
    if (mask.side    == a.mask.side    && mask.side    == true && side    != a.side)    return false;
    if (mask.station == a.mask.station && mask.station == true && station != a.station) return false;
    if (mask.ring    == a.mask.ring    && mask.ring    == true && ring    != a.ring)    return false;
    if (mask.chamber == a.mask.chamber && mask.chamber == true && chamber != a.chamber) return false;
    if (mask.layer   == a.mask.layer   && mask.layer   == true && layer   != a.layer)   return false;
    if (mask.cfeb    == a.mask.cfeb    && mask.cfeb    == true && cfeb    != a.cfeb)    return false;
    if (mask.hv      == a.mask.hv      && mask.hv      == true && hv      != a.hv)      return false;
    return true;
  };

  const Address* operator= (const Address& a) {
    mask.side    = a.mask.side;
    side         = a.side;
    mask.station = a.mask.station;
    station      = a.station;
    mask.ring    = a.mask.ring;
    ring         = a.ring;
    mask.chamber = a.mask.chamber;
    chamber      = a.chamber;
    mask.layer   = a.mask.layer;
    layer        = a.layer;
    mask.cfeb    = a.mask.cfeb;
    cfeb         = a.cfeb;
    mask.hv      = a.mask.hv;
    hv           = a.hv;
    return this;
  };

};

/**
 * @brief  Area covered by Address in eta/phy space
 */
struct AddressBox {
  Address adr;
  float xmin;
  float xmax;
  float ymin;
  float ymax;
};

/** Map of partitions and partition covering adresses indexes type */
typedef std::map<unsigned int, std::vector<unsigned int> > PartitionMap;

/** Iterator type of PartitionMap */
typedef PartitionMap::iterator PartitionMapIterator;

/**
 * @class Detector
 * @brief Detector geometry and addressing related imformation and routines
 */
class Detector {

  public:

    Detector(unsigned int p_partitions_x = 0, unsigned int p_partitions_y = 0);

    bool NextAddress(unsigned int& i, const Address*& adr, const Address& mask) const;
    bool NextAddressBox(unsigned int& i, const AddressBox*& box, const Address& mask) const;
    //bool NextAddressBoxByPartition(unsigned int& i, unsigned int& px, unsigned int& py, const AddressBox*& box, const Address& mask, float xmin, float xmax, float ymin, float ymax);
    bool NextAddressBoxByPartition (unsigned int& i, unsigned int px, unsigned int py, AddressBox*& box);

    float Area(unsigned int station) const;
    float Area(const Address& adr) const;

    void PrintAddress(const Address& adr) const;
    const std::string AddressName(const Address& adr) const;
    bool AddressFromString(const std::string str_address, Address& adr) const;

    unsigned int NumberOfRings(unsigned int station) const;
    unsigned int NumberOfChambers(unsigned int station, unsigned int ring) const;
    unsigned int NumberOfChamberCFEBs(unsigned int station, unsigned int ring) const;
    unsigned int NumberOfChamberHVs(unsigned int station, unsigned int ring) const;
    int NumberOfChamberParts(int station, int ring) const;
    std::string ChamberPart(int npart) const;
    bool isChamberInstalled(int side, int station, int ring, int chamber) const;
    int SideSign(int side) const;
    int NumberOfWiregroups(int station, int ring ) const;

    // Methods to find relative position of chambers w.r.t. CMS global coordinate system
    double RPin(int station, int ring) const;
    double PhiDegChamberCenter(int station, int ring, int chamber) const;
    double PhiRadChamberCenter(int station, int ring, int chamber) const {
      return PI/180.0*PhiDegChamberCenter(station, ring, chamber);
    };

    // Wire Groups
    double LocalYtoBeam(int side, int station, int ring, int wgroup) const;
    double LocalYtoBeam(int side, int station, int ring, const std::string &part, int hstrip, int wgroup) const;

    // strips and hstrips
    double stripStaggerInstripWidth(int station, int ring, int layer) const;

    int NumberOfStrips(int station, int ring, const std::string &part) const;
    int NumberOfStrips(int station, int ring) const { return NumberOfStrips( station, ring, "b" ); };
    int NumberOfHalfstrips(int station, int ring, const std::string &part) const { return 2 * NumberOfStrips(station, ring, part); };
    int NumberOfHalfstrips(int station, int ring) const { return NumberOfHalfstrips( station, ring, "b" ); };

    double stripDPhiDeg(int station, int ring, const std::string &part) const;
    double stripDPhiDeg(int station, int ring ) const { return stripDPhiDeg(station, ring, "b"); };
    double stripDPhiRad(int station, int ring, const std::string &part) const { return PI/180.0*stripDPhiDeg( station, ring, part ); };
    double stripDPhiRad(int station, int ring ) const { return stripDPhiRad(station, ring, "b"); };

    double hstripDPhiDeg(int station, int ring, const std::string &part) const { return 0.5*stripDPhiDeg( station, ring, part ); };
    double hstripDPhiDeg(int station, int ring) const { return hstripDPhiDeg( station, ring, "b" ); };
    double hstripDPhiRad(int station, int ring, const std::string &part) const { return PI/180.0*hstripDPhiDeg( station, ring, part ); };
    double hstripDPhiRad(int station, int ring) const { return hstripDPhiRad( station, ring, "b" ); };

    double LocalPhiDegStripToChamberCenter(int side, int station, int ring, const std::string &part, int layer, int strip) const;
    double LocalPhiDegStripToChamberCenter(int side, int station, int ring, int layer, int strip) const {
      return LocalPhiDegStripToChamberCenter(side, station, ring, "b", layer, strip);
    };

    double LocalPhiRadStripToChamberCenter(int side, int station, int ring, const std::string &part, int layer, int strip) const {
      return PI/180.0*LocalPhiDegStripToChamberCenter(side, station, ring, part, layer, strip);
    };
    double LocalPhiRadStripToChamberCenter(int side, int station, int ring, int layer, int strip) const {
      return PI/180.0*LocalPhiDegStripToChamberCenter(side, station, ring, layer, strip);
    };

    double LocalPhiDegHstripToChamberCenter(int side, int station, int ring, const std::string &part, int layer, int hstrip) const;
    double LocalPhiDegHstripToChamberCenter(int side, int station, int ring, int layer, int hstrip) const {
      return LocalPhiDegHstripToChamberCenter(side, station, ring, "b", layer, hstrip);
    };

    double LocalPhiRadHstripToChamberCenter(int side, int station, int ring, const std::string &part, int layer, int hstrip) const {
      return PI/180.0*LocalPhiDegHstripToChamberCenter(side, station, ring, part, layer, hstrip);
    };
    double LocalPhiRadHstripToChamberCenter(int side, int station, int ring, int layer, int hstrip) const {
      return PI/180.0*LocalPhiDegHstripToChamberCenter(side, station, ring, layer, hstrip);
    };

    double Z_mm(int side, int station, int ring, int chamber, int layer) const;

    double Phi_deg(int side, int station, int ring, const std::string &part, int chamber, int layer, int hstrip) const;
    double Phi_rad(int side, int station, int ring, const std::string &part, int chamber, int layer, int hstrip) const {
      return PI/180.0*Phi_deg(side, station, ring, part, chamber, layer, hstrip);
    };

    double R_mm(int side, int station, int ring, const std::string &part, int layer, int hstrip, int wgroup) const;

    double X_mm(int side, int station, int ring, const std::string &part, int chamber, int layer, int hstrip, int wgroup) const;
    double Y_mm(int side, int station, int ring, const std::string &part, int chamber, int layer, int hstrip, int wgroup) const;

    double Theta_rad(int side, int station, int ring, const std::string &part, int chamber, int layer, int hstrip, int wgroup) const;
    double Theta_deg(int side, int station, int ring, const std::string &part, int chamber, int layer, int hstrip, int wgroup) const {
      return 180.0/PI*Theta_rad(side, station, ring, part, chamber, layer, hstrip, wgroup);
    };

    void chamberBoundsXY(int side, int station, int ring, int chamber, const std::string& part, double* x, double* y) const;

  private:

    float Eta(float r, float z) const;
    float EtaToX(float eta) const;
    float PhiToY(float phi) const;
    float Z(const int station, const int ring) const;
    float RMinHV(const int station, const int ring, const int n_hv) const;
    float RMaxHV(const int station, const int ring, const int n_hv) const;
    float PhiMinCFEB(const int station, const int ring, const int chamber, const int cfeb) const;
    float PhiMaxCFEB(const int station, const int ring, const int chamber, const int cfeb) const;
    void chamberBoundsXY(int side, int station, int ring, int chamber, const std::string& part, int hs, int wg, double& x, double& y) const;

    /** Address boxes in epa/phi space */
    AddressBox boxes[N_ELEMENTS];

    /** Station areas precalculated */
    float station_area[N_STATIONS];

    /** Number of partitions in X axis */
    unsigned int partitions_x;

    /** Number of partitions in Y axis */
    unsigned int partitions_y;

    /** Map of partitions and list of it covering addresses indexes */
    PartitionMap partitions;

};

}

#endif
