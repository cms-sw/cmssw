#ifndef RecoAlgos_SingleObjectSelector_h
#define RecoAlgos_SingleObjectSelector_h
/** \class SingleTrackSelector
 *
 * selects a subset of a collection based
 * on single track selection done via functor
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 * $Id: SingleTrackSelector.h,v 1.3 2006/07/21 14:11:26 llista Exp $
 *
 */

#include "PhysicsTools/RecoAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/RecoAlgos/interface/SingleElementCollectionSelector.h"

template<typename C, typename S>
class SingleObjectSelector : 
  public ObjectSelector<C, SingleElementCollectionSelector<C, S> > {
public:
  SingleObjectSelector( const edm::ParameterSet & cfg ) : 
    ObjectSelector<C, SingleElementCollectionSelector<C, S> >( cfg ) { }
  ~SingleObjectSelector() { }
};

#endif
