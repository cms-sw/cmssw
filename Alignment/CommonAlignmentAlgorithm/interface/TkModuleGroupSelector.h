#ifndef Alignment_CommonAlignmentAlgorithm_TkModuleGroupSelector_h
#define Alignment_CommonAlignmentAlgorithm_TkModuleGroupSelector_h

/**
 * \file TkModuleGroupSelector.cc
 *
 * Class provides an algorithm which assigns
 * runrange-dependent (IOV-dependent)
 * indices to groups of tracker modules.
 *
 *  \author Joerg Behr
 *  \date May 2013
 *  $Revision: 1.2 $
 *  $Date: 2013/05/31 12:13:40 $
 *  (last update by $Author: flucke $)
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

#include <vector>
#include <map>
#include <list>


class AlignableTracker;
class AlignableMuon;
class AlignableExtras;

namespace edm { class EventSetup; class ParameterSet; } 

class TkModuleGroupSelector
{
public:
  /// Constructor
  explicit TkModuleGroupSelector(AlignableTracker *aliTracker,
                                 const edm::ParameterSet &cfg,
                                 const std::vector<int> &sdets);
  
  /// Destructor
  virtual ~TkModuleGroupSelector() {};

  // Returns the number of parameters.
  unsigned int getNumberOfParameters() const;

  /// Total number of IOVs.
  unsigned int numIovs() const;

  /// First run of iov (0 if iovNum not treated).
  edm::RunNumber_t firstRunOfIOV(unsigned int iovNum) const;

  /// Index of parameter for given detId (detId not treated => < 0)
  /// and the given run.
  int getParameterIndexFromDetId(unsigned int detId, edm::RunNumber_t run) const;
  
 private:
  // Constructs the run-dependent module groups from configuration parameters.
  void createModuleGroups(AlignableTracker *aliTracker,
                          const edm::VParameterSet &granularityConfig,
                          const std::vector<edm::RunNumber_t> &defaultRunRange,
                          edm::RunNumber_t defaultReferenceRun);
                    
  // Method used to test the provided configuration for unknown parameters
  void verifyParameterNames(const edm::ParameterSet &pset, unsigned int psetnr) const;
  
  // Method to test whether the split option has been turned on
  const bool testSplitOption(const edm::ParameterSet &pset) const;

  // Add modules to a specific group which is also created in this function.
  bool createGroup(
                   unsigned int &Id, //id of the first run
                   const std::vector<edm::RunNumber_t> &range, //run range
                   const std::list<Alignable*> &selected_alis, //list of alignables for which a group is created
                   const edm::RunNumber_t refrun //reference run number
                   );
  
  // Fill the container which is a map between the det id and the id of the group
  // to which the module belongs.
  void fillDetIdMap(const unsigned int detid, const unsigned int groupid);

  // Array with run boundaries which is a combination
  // of all defined run ranges of all specified module groups.
  std::vector<edm::RunNumber_t> globalRunRange_;

  // For a given module group the id of the first IOV.
  std::vector<unsigned int> firstId_;

  // Run range per module group
  std::vector<std::vector<edm::RunNumber_t> > runRange_;

  // Mapping between module id and module group id.
  std::map<unsigned int, unsigned int> mapDetIdGroupId_;

  // Total number of parameters.
  unsigned int nparameters_;

  // The ids of the subdetectors for which parameters are determined.
  std::vector<int> subdetids_;

  // Reference run per module group
  std::vector<edm::RunNumber_t> referenceRun_;
  
};

#endif
