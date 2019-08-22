#ifndef CondFormats_CSCTriggerMapping_h
#define CondFormats_CSCTriggerMapping_h

/** 
 * \class CSCTriggerMapping
 * \author Lindsey Gray (taken from T. Cox's design)
 * Abstract class to define mapping between CSC Trigger Hardware labels and
 * geometry labels. Basically this amounts to a cabling scheme.
 *
 * Defines the ids and labels in the mapping and supplies tramslation interface.
 * A derived class must define how hardware labels map to a unique integer.
 * A derived, concrete, class must define from where the mapping information comes.
 */

//@@ FIXME This whole design would better suit a Factory/Builder pattern

#include "CondFormats/Serialization/interface/Serializable.h"

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <vector>
#include <map>

class CSCTriggerMapping {
public:
  /// Default constructor
  CSCTriggerMapping();

  /// Destructor
  virtual ~CSCTriggerMapping();

  /**
   * Instead of a set of vectors of int use one vector of a set of ints
   * Defines a connection between a chamber on a disc and a readout label. 
   * This is equivalent to the placement of a board in a crate, and a MPC to SR/SP 
   * optical connection.
   * Construction of CSCDetIds is done using CSCTriggerNumbering.
   * 
   * variables with a 'r' prefix are readout-derived labels
   * variables with a 'c' prefix are geometry-derived labels (c as in chamber label)
   * \warning ALL LABELS ARE TRIGGER LABELS. PLEASE ACCOUNT FOR THIS!!!
   */
  typedef struct CSCTriggerConnection {
    CSCTriggerConnection() {}
    CSCTriggerConnection(int rendcap,
                         int rstation,
                         int rsector,
                         int rsubsector,
                         int rcscid,
                         int cendcap,
                         int cstation,
                         int csector,
                         int csubsector,
                         int ccscid)
        : rendcap_(rendcap),
          rstation_(rstation),
          rsector_(rsector),
          rsubsector_(rsubsector),
          rcscid_(rcscid),
          cendcap_(cendcap),
          cstation_(cstation),
          csector_(csector),
          csubsector_(csubsector),
          ccscid_(ccscid) {}
    ~CSCTriggerConnection() {}

    int rendcap_;
    int rstation_;
    int rsector_;
    int rsubsector_;
    int rcscid_;
    int cendcap_;
    int cstation_;
    int csector_;
    int csubsector_;
    int ccscid_;

    COND_SERIALIZABLE;
  } Connection;

  /**
    * Return CSCDetId for chamber/layer corresponding to readout ids station, sector, subsector and 
    * cscid for given endcap and layer no. 1-6, or for chamber if no layer no. supplied.
    * Args: endcap = 1 (+z), 2 (-z), station, readout sector, readout subsector, readout cscid, layer#
    */
  // layer at end so it can have default arg
  CSCDetId detId(int endcap, int station, int sector, int subsector, int cscid, int layer = 0) const;

  /** 
    * Return chamber label corresponding to readout ids station, sector, subsector and cscid for given endcap
    *  endcap = 1 (+z), 2 (-z), station, sector, subsector, cscid (dmb slot/2)
    */
  int chamber(int endcap, int station, int sector, int subsector, int cscid) const;

  /** 
    * Fill mapping store
    */
  virtual void fill(void) = 0;

  /**
    * Add one record of info to mapping
    */
  void addRecord(int rendcap,
                 int rstation,
                 int rsector,
                 int rsubsector,
                 int rcscid,
                 int cendcap,
                 int cstation,
                 int csector,
                 int csubsector,
                 int ccscid);

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
     */
  virtual int hwId(int endcap, int station, int sector, int subsector, int cscid) const = 0;

  /**
     * Build a unique integer out of chamber labels.
     *
     * Translate to geometry labels then use rawId.
     */
  int swId(int endcap, int station, int sector, int subsector, int cscid) const;

  std::string myName_ COND_TRANSIENT;
  bool debugV_ COND_TRANSIENT;
  std::vector<Connection> mapping_;
  std::map<int, int> hw2sw_ COND_TRANSIENT;

  COND_SERIALIZABLE;
};

#endif
