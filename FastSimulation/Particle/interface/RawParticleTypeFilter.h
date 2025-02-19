#ifndef RAWPARTTICLETYPEFILTER_H
#define RAWPARTTICLETYPEFILTER_H

#include "FastSimulation/Particle/interface/BaseRawParticleFilter.h"
#include "FastSimulation/Particle/interface/RawParticle.h"

/**
 * A RawParticle filter class.
 *  This class rejects or accepts RawParticles based on their pid().
 *  It allows to either accept or (exclusive) reject particle types.
 * \author Stephan Wynhoff
 */
class RawParticleTypeFilter : public BaseRawParticleFilter {
public:

  RawParticleTypeFilter(){ ; };

  /// Construct a filter to accept particles of \a particleName
  RawParticleTypeFilter(const std::string& particleName);

  /// Construct a filter to accept particles of \a particleName1 or \a particleName2
  RawParticleTypeFilter(const std::string& particleName1, 
			const std::string& particleName2);

  /// Construct a filter to accept particles with id \a pid
  RawParticleTypeFilter(const int pid);

  /// Construct a filter to accept particles with id \a pid1 or \a pid2
  RawParticleTypeFilter(const int pid1, const int pid2);

  virtual ~RawParticleTypeFilter(){;};

public:

  /** Accept in addition particles with id \a id. <b> The list of
   *  particles to reject will be cleared and no longer used.</b> */
  void addAccept(const int id);

  /** Accept in addition particles with name \a name. <b> The list of
   *  particles to reject will be cleared and no longer used.</b> */
  void addAccept(const std::string& name);

  /** Reject in addition particles with id \a id. <b> The list of
   *  particles to accept will be cleared and no longer used.</b> */
  void addReject(const int id);

  /** Reject in addition particles with name \a name. <b> The list of
   *  particles to accept will be cleared and no longer used.</b> */
  void addReject(const std::string& name);

private:

  /// implemented as required by class BaseRawParticleFilter
  bool isOKForMe(const RawParticle *p) const;

  /// is this particle id in the list of acceptable particles?
  bool isAcceptable(const int id) const;

  /// is this particle id in the list of rejectable particles?
  bool isRejectable(const int id) const;

private:
  std::vector<int> myAcceptIDs, myRejectIDs;
};

#endif
