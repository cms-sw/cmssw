#ifndef L1Trigger_CSCSectorReceiverLUT_h
#define L1Trigger_CSCSectorReceiverLUT_h

/**
 * \class CSCSectorReceiverLUT
 * \author Lindsey Gray
 *
 * Provides Look Up Table information for use in the SP Core.
 * Partial port from ORCA.
 */

#include <FWCore/ParameterSet/interface/ParameterSet.h>

class CSCSectorReceiverLUT
{
 public:

  CSCSectorReceiverLUT(int endcap, int sector, int subsector, int station, const edm::ParameterSet &pset);
  ~CSCSectorReceiverLUT();

  // Address Types
  typedef struct local_phi_address
  {
    unsigned int strip        : 8;
    unsigned int clct_pattern : 3;
    unsigned int pattern_type : 1; // 1 is half strip 0 is di strip
    unsigned int quality      : 4;
    unsigned int lr           : 1;
    unsigned int spare        : 2;
  } lclphiadd;

  typedef struct global_phi_address
  {
    unsigned int phi_local    : 9;
    unsigned int wire_group   : 5;  // bits 2-6 of wg
    unsigned int cscid        : 4;
  } gblphiadd;

  typedef struct global_eta_address
  {
    unsigned int phi_bend     : 6;
    unsigned int phi_local    : 2;
    unsigned int wire_group   : 7;
    unsigned int cscid        : 4;
  } gbletaadd;

  /// Data Types
  typedef struct local_phi_data
  {
    unsigned short phi_local      : 10;
    unsigned short phi_bend_local : 6;
  } lclphidat;

  typedef struct global_phi_data
  {
    unsigned short global_phi : 12;
    unsigned short spare      : 4;
  } gblphidat;

  typedef struct global_eta_data
  {
    unsigned short global_eta  : 7;
    unsigned short global_bend : 5;
    unsigned short spare       : 4;
  } gbletadat;

  ///Geometry Lookup Tables

  lclphidat localPhi(int strip, int pattern, int quality, int lr) const;
  lclphidat localPhi(unsigned address) const;
  lclphidat localPhi(lclphiadd address) const;

  /*
  unsigned short globalPhiME(int phi_local, int wire_group, int cscid) const;
  unsigned short globalPhiME(unsigned address) const;

  unsigned short globalPhiMB(int phi_local,int wire_group, int cscid) const;
  unsigned short globalPhiMB(unsigned address) const;
  */

  gbletadat globalEtaME(int phi_bend, int phi_local, int wire_group, int cscid) const;
  gbletadat globalEtaME(unsigned address) const;
  gbletadat globalEtaME(gbletaadd address) const;

  /// Helpers
  std::string encodeFileIndex() const;

 private:
  int _endcap, _sector, _subsector, _station;

  /// Local Phi LUT 
  lclphidat calcLocalPhi(const lclphiadd& address) const;
  
  /// Global Eta LUT
  gbletadat calcGlobalEtaME(const gbletaadd& address) const;
  double getEtaValue(const unsigned& cscid, const unsigned& wire_group, const unsigned& phi_local) const;

  void fillLocalPhiLUT();
  
  bool LUTsFromFile;

  /// Arrays for holding read in LUT information.
  /// MB LUT arrays only initialized in ME1
  void readLUTsFromFile();
  
  static bool me_lcl_phi_loaded;
  static lclphidat* me_lcl_phi;
  //gblphidat* me_global_phi, mb_global_phi;
  gbletadat* me_global_eta;
};

#endif
