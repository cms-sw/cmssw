#ifndef CondFormats_CSCReadoutMapping_h
#define CondFormats_CSCReadoutMapping_h

/** 
 * \class CSCReadoutMapping
 * \author Tim Cox
 * Abstract class to define mapping between CSC readout hardware ids and other labels.
 *
 * Defines the ids and labels in the mapping and supplies translation interface.
 * A derived class must define how hardware labels map to a unique integer.
 * A derived, concrete, class must define from where the mapping information comes.
 */

//@@ FIXME This whole design would better suit a Factory/Builder pattern

#include "CondFormats/Serialization/interface/Serializable.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>
#include <map>

class CSCReadoutMapping {
public:
  /// Default constructor
  CSCReadoutMapping();

  /// Destructor
  virtual ~CSCReadoutMapping();

  /**
   * Instead of a set of vectors of int use one vector of a set of ints
   */
  struct CSCLabel {
    CSCLabel() {}
    CSCLabel(int endcap,
             int station,
             int ring,
             int chamber,
             int vmecrate,
             int dmb,
             int tmb,
             int tsector,
             int cscid,
             int ddu,
             int dcc)
        : endcap_(endcap),
          station_(station),
          ring_(ring),
          chamber_(chamber),
          vmecrate_(vmecrate),
          dmb_(dmb),
          tmb_(tmb),
          tsector_(tsector),
          cscid_(cscid),
          ddu_(ddu),
          dcc_(dcc) {}
    ~CSCLabel() {}

    int endcap_;
    int station_;
    int ring_;
    int chamber_;
    int vmecrate_;
    int dmb_;
    int tmb_;
    int tsector_;
    int cscid_;
    int ddu_;
    int dcc_;

    COND_SERIALIZABLE;
  };

  /**
    * Return CSCDetId for layer corresponding to readout ids vme, tmb, and dmb for given endcap
    * and layer no. 1-6, or for chamber if no layer no. supplied.
    * Args: endcap = 1 (+z), 2 (-z), station, vme crate number, dmb slot number, tmb slot number, 
    * cfeb number (so we can identify ME1a/b within ME11), layer number
    */
  // layer at end so it can have default arg
  CSCDetId detId(int endcap, int station, int vmecrate, int dmb, int tmb, int cfeb, int layer = 0) const;

  /** 
    * Return chamber label corresponding to readout ids vme, tmb and dmb for given endcap
    *  endcap = 1 (+z), 2 (-z), station, vme crate number, dmb slot number, tmb slot number.
    */
  int chamber(int endcap, int station, int vmecrate, int dmb, int tmb) const;

  ///returns hardware ids given chamber id
  CSCLabel findHardwareId(const CSCDetId&) const;
  ///returns vmecrate given CSCDetId
  int crate(const CSCDetId&) const;
  ///returns dmbId given CSCDetId
  int dmbId(const CSCDetId&) const;
  ///returns DCC# given CSCDetId
  int dccId(const CSCDetId&) const;
  ///returns DDU# given CSCDetId
  int dduId(const CSCDetId&) const;

  /**
    * Add one record of info to mapping
    */
  void addRecord(int endcap,
                 int station,
                 int ring,
                 int chamber,
                 int vmecrate,
                 int dmb,
                 int tmb,
                 int tsector,
                 int cscid,
                 int ddu,
                 int dcc);

  /**
     * Set debug printout flag
     */
  void setDebugV(bool dbg) { debugV_ = dbg; }

  /**
     * Status of debug printout flag
     */
  bool debugV(void) const { return debugV_; }

  /**
     * Return class name
     */
  const std::string& myName(void) const { return myName_; }

private:
  /**
     * Build a unique integer out of the readout electronics labels.
     *
     * In general this must depend on endcap and station, as well as
     * vme crate number and dmb slot number. And possibly tmb slot?
     */
  virtual int hwId(int endcap, int station, int vme, int dmb, int tmb) const = 0;

  /**
     * Build a unique integer out of chamber labels.
     *
     * We'll probably use rawId of CSCDetId... You know it makes sense!
     */
  int swId(int endcap, int station, int ring, int chamber) const;

  std::string myName_ COND_TRANSIENT;
  bool debugV_ COND_TRANSIENT;
  std::vector<CSCLabel> mapping_;
  std::map<int, int> hw2sw_ COND_TRANSIENT;
  std::map<int, CSCLabel> sw2hw_;

  COND_SERIALIZABLE;
};

#endif
