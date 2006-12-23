/** \file CompositeAlignmentParameters.cc
 *
 *  $Date: 2006/10/19 14:20:59 $
 *  $Revision: 1.3 $
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignmentParametrization/interface/CompositeAlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/CompositeAlignmentDerivativesExtractor.h"


//__________________________________________________________________________________________________
CompositeAlignmentParameters::
CompositeAlignmentParameters(const AlgebraicVector& par, 
			     const AlgebraicSymMatrix& cov, const Components& comp) :
  AlignmentParameters(0,par,cov) ,
  theComponents(comp) 
{}


//__________________________________________________________________________________________________
CompositeAlignmentParameters::
CompositeAlignmentParameters(const AlgebraicVector& par, 
			     const AlgebraicSymMatrix& cov, const Components& comp, 
			     const AlignableDetToAlignableMap& map,
			     const Aliposmap& aliposmap,
			     const Alilenmap& alilenmap) :
  AlignmentParameters(0,par,cov) ,
  theComponents(comp) ,
  theAlignableDetToAlignableMap(map),
  theAliposmap(aliposmap),
  theAlilenmap(alilenmap)
{}


//__________________________________________________________________________________________________
CompositeAlignmentParameters::
CompositeAlignmentParameters(const DataContainer& data, const Components& comp, 
			     const AlignableDetToAlignableMap& map,
			     const Aliposmap& aliposmap,
			     const Alilenmap& alilenmap) :
  AlignmentParameters(0,data) ,
  theComponents(comp) ,
  theAlignableDetToAlignableMap(map),
  theAliposmap(aliposmap),
  theAlilenmap(alilenmap)
{}


//__________________________________________________________________________________________________
CompositeAlignmentParameters::~CompositeAlignmentParameters() 
{}


//__________________________________________________________________________________________________
CompositeAlignmentParameters* 
CompositeAlignmentParameters::clone( const AlgebraicVector& par, 
				     const AlgebraicSymMatrix& cov) const
{
  
  CompositeAlignmentParameters* cap = 
    new CompositeAlignmentParameters(par,cov,components());

  if ( userVariables() )
    cap->setUserVariables(userVariables()->clone());

  return cap;

}


//__________________________________________________________________________________________________
CompositeAlignmentParameters* 
CompositeAlignmentParameters::cloneFromSelected( const AlgebraicVector& par, 
						 const AlgebraicSymMatrix& cov) const
{

  return clone(par,cov);

}


//__________________________________________________________________________________________________
CompositeAlignmentParameters* 
CompositeAlignmentParameters::clone( const AlgebraicVector& par, 
				     const AlgebraicSymMatrix& cov,
				     const AlignableDetToAlignableMap& map, 
				     const Aliposmap& aliposmap,
				     const Alilenmap& alilenmap ) const
{

  CompositeAlignmentParameters* cap = 
    new CompositeAlignmentParameters(par,cov,components(),map,aliposmap,alilenmap);

  if ( userVariables() )
    cap->setUserVariables(userVariables()->clone());

  return cap;

}


//__________________________________________________________________________________________________
CompositeAlignmentParameters* 
CompositeAlignmentParameters::cloneFromSelected( const AlgebraicVector& par, 
						 const AlgebraicSymMatrix& cov, 
						 const AlignableDetToAlignableMap& map, 
						 const Aliposmap& aliposmap,
						 const Alilenmap& alilenmap) const
{

  return clone(par,cov,map,aliposmap,alilenmap);

}


//__________________________________________________________________________________________________
CompositeAlignmentParameters::Components 
CompositeAlignmentParameters::components() const
{ 
  return theComponents;
}


//__________________________________________________________________________________________________
// full derivatives for a composed object
AlgebraicMatrix
CompositeAlignmentParameters::derivatives( const std::vector<TrajectoryStateOnSurface>& tsosvec,
					   const std::vector<AlignableDet*>& alidetvec ) const
{
  std::vector<Alignable*> alivec;
  for (std::vector<AlignableDet*>::const_iterator it=alidetvec.begin();
	   it!=alidetvec.end(); ++it)
	alivec.push_back(alignableFromAlignableDet(*it));
  
  CompositeAlignmentDerivativesExtractor theExtractor(alivec,alidetvec,tsosvec);
  return theExtractor.derivatives();
}

//__________________________________________________________________________________________________
AlgebraicVector 
CompositeAlignmentParameters::correctionTerm( const std::vector<TrajectoryStateOnSurface>& tsosvec,
					      const std::vector<AlignableDet*>& alidetvec) const
{
  std::vector<Alignable*> alivec;
  for (std::vector<AlignableDet*>::const_iterator it=alidetvec.begin();
	   it!=alidetvec.end(); ++it )
	alivec.push_back(alignableFromAlignableDet(*it));
  
  CompositeAlignmentDerivativesExtractor theExtractor(alivec,alidetvec,tsosvec);
  return theExtractor.correctionTerm();
}
 	 
//__________________________________________________________________________________________________ 	 
// assume all are selected
AlgebraicMatrix CompositeAlignmentParameters::
selectedDerivatives( const std::vector<TrajectoryStateOnSurface>& tsosvec,
		     const std::vector<AlignableDet*>& alidetvec) const
{ 
  return derivatives(tsosvec,alidetvec);
}

//__________________________________________________________________________________________________ 	 
// only one (tsos,AlignableDet) as argument [for compatibility with base class]
AlgebraicMatrix 
CompositeAlignmentParameters::derivatives( const TrajectoryStateOnSurface &tsos, 
					   AlignableDet* alidet) const
{
  std::vector<TrajectoryStateOnSurface> tsosvec;
  std::vector<AlignableDet*> alidetvec;
  tsosvec.push_back(tsos);
  alidetvec.push_back(alidet);
  return derivatives(tsosvec,alidetvec);
}
 	
//__________________________________________________________________________________________________ 
// assume all are selected
AlgebraicMatrix 
CompositeAlignmentParameters::selectedDerivatives( const TrajectoryStateOnSurface &tsos, 
						   AlignableDet* alidet ) const
{ 
  return derivatives(tsos,alidet);
}
 	 

// Derivatives ----------------------------------------------------------------
// legacy methods
// full derivatives for a composed object
AlgebraicMatrix CompositeAlignmentParameters::
derivativesLegacy( const std::vector<TrajectoryStateOnSurface> &tsosvec, 
		   const std::vector<AlignableDet*>& alidetvec ) const
{

  // sanity check: length of parameter argument vectors must be equal
  if (alidetvec.size() != tsosvec.size()) {
	edm::LogError("BadArgument") << " Inconsistent length of argument vectors! ";
    AlgebraicMatrix selderiv(1,0);
    return selderiv;
  }

  std::vector<AlgebraicMatrix> vecderiv;
  int nparam=0;

  std::vector<TrajectoryStateOnSurface>::const_iterator itsos=tsosvec.begin();
  for( std::vector<AlignableDet*>::const_iterator it=alidetvec.begin(); 
	   it!=alidetvec.end(); ++it, ++itsos ) 
	{
	  AlignableDet* ad = (*it);
	  Alignable*    ali = alignableFromAlignableDet(ad);
	  AlignmentParameters* ap = ali->alignmentParameters();
	  AlgebraicMatrix thisselderiv = ap->selectedDerivatives(*itsos,ad);
	  vecderiv.push_back(thisselderiv);
	  nparam += thisselderiv.num_row();
	}

  int ipos=1;
  AlgebraicMatrix selderiv(nparam,2);
  for ( std::vector<AlgebraicMatrix>::const_iterator imat=vecderiv.begin();
		imat!=vecderiv.end(); ++imat ) 
	{
	  AlgebraicMatrix thisselderiv=(*imat);
	  int npar=thisselderiv.num_row();
	  selderiv.sub(ipos,1,thisselderiv);
	  ipos += npar;
	}

  return selderiv;
}


//__________________________________________________________________________________________________
// assume all are selected
AlgebraicMatrix CompositeAlignmentParameters::
selectedDerivativesLegacy( const std::vector<TrajectoryStateOnSurface> &tsosvec, 
			   const std::vector<AlignableDet*>& alidetvec ) const
{ 
  return derivativesLegacy(tsosvec,alidetvec);
}


//__________________________________________________________________________________________________
// only one (tsos,AlignableDet) as argument [for compatibility with base class]
AlgebraicMatrix 
CompositeAlignmentParameters::derivativesLegacy( const TrajectoryStateOnSurface& tsos, 
						 AlignableDet* alidet ) const
{

  std::vector<TrajectoryStateOnSurface> tsosvec;
  std::vector<AlignableDet*> alidetvec;
  tsosvec.push_back(tsos);
  alidetvec.push_back(alidet);
  return derivativesLegacy(tsosvec,alidetvec);

}


//__________________________________________________________________________________________________
// assume all are selected
AlgebraicMatrix 
CompositeAlignmentParameters::selectedDerivativesLegacy( const TrajectoryStateOnSurface& tsos, 
							 AlignableDet* alidet ) const
{ 
  return derivativesLegacy(tsos,alidet);
}


//__________________________________________________________________________________________________
// finds Alignable corresponding to AlignableDet
Alignable* 
CompositeAlignmentParameters::alignableFromAlignableDet(AlignableDet* adet) const
{

  AlignableDetToAlignableMap::const_iterator iali =
    theAlignableDetToAlignableMap.find(adet);
  if ( iali!=theAlignableDetToAlignableMap.end() ) return (*iali).second;
  else return 0;

}


//__________________________________________________________________________________________________
AlgebraicVector
CompositeAlignmentParameters::parameterSubset( const std::vector<Alignable*>& veci ) const
{

  int ndim=0;
  // iterate over input vector of alignables to determine size of result vector
  for ( std::vector<Alignable*>::const_iterator it=veci.begin();
		it != veci.end(); ++it) 
	{

	  // check if in components 
	  std::vector<Alignable*>::const_iterator ifind = 
		std::find( theComponents.begin(), theComponents.end(), *it );
	  if ( ifind == theComponents.end() ) 
		{ 
		  edm::LogError("NotFound") << "Alignable not found in components!";
		  return AlgebraicVector();
		}

	  // get pos/length
	  Aliposmap::const_iterator iposmap = theAliposmap.find( *it );
	  Alilenmap::const_iterator ilenmap = theAlilenmap.find( *it );
	  if ( iposmap == theAliposmap.end() || ilenmap != theAlilenmap.end() )
		{
		  edm::LogError("NotFound") << "pos,len not found for Ali in maps!";
		  return AlgebraicVector();
		}

	  ndim += (*ilenmap).second;

	}

  AlgebraicVector result(ndim,0);
  int ires=1;

  // now iterate again to do the actual work...
  for ( std::vector<Alignable*>::const_iterator it=veci.begin();
		it!=veci.end(); ++it ) 
	{
	  Aliposmap::const_iterator iposmap=theAliposmap.find( *it );
	  Alilenmap::const_iterator ilenmap=theAlilenmap.find( *it );
	  int pos=(*iposmap).second;
	  int len=(*ilenmap).second;
	  AlgebraicVector piece = theData->parameters().sub(pos,pos+len-1);
	  result.sub( ires, piece );
	  ires += len;
	}

  return result;

}


//__________________________________________________________________________________________________
// extract covariance between two subsets of alignables
AlgebraicMatrix 
CompositeAlignmentParameters::covarianceSubset( const std::vector<Alignable*>& veci, 
						const std::vector<Alignable*>& vecj ) const
{

  int ndimi=0;
  int ndimj=0;

  // iterate over input vectors of alignables
  // to determine dimensions of result matrix
  for ( std::vector<Alignable*>::const_iterator it=veci.begin(); 
		it != veci.end(); ++it ) 
	{
	  // check if in components 
	  std::vector<Alignable*>::const_iterator ifind = std::find( theComponents.begin(),
								     theComponents.end(), *it );
	  if ( ifind == theComponents.end() ) 
		{
		  edm::LogError("NotFound") << "Alignable not found in components!";
		  return AlgebraicMatrix();
		}

	  // get pos/length
	  Aliposmap::const_iterator iposmap = theAliposmap.find( *it );
	  Alilenmap::const_iterator ilenmap = theAlilenmap.find( *it );
	  if ( iposmap == theAliposmap.end() || ilenmap == theAlilenmap.end() ) 
		{
		  edm::LogError("NotFound") << "pos,len not found for Ali in maps!";
		  return AlgebraicMatrix();
		}
	  ndimi += (*ilenmap).second;
	}

  // vector vecj
  for ( std::vector<Alignable*>::const_iterator it=vecj.begin(); 
		it != vecj.end(); ++it ) 
	{
	  // check if in components 
	  std::vector<Alignable*>::const_iterator ifind = std::find( theComponents.begin(),
								     theComponents.end(), *it );
	  if (ifind == theComponents.end()) 
		{ 
		  edm::LogError("NotFound") << "Alignable not found in components!";
		  return AlgebraicMatrix();
		}

	  // get pos/length
	  Aliposmap::const_iterator iposmap = theAliposmap.find( *it );
	  Alilenmap::const_iterator ilenmap = theAlilenmap.find( *it );
	  if (iposmap == theAliposmap.end() || ilenmap == theAlilenmap.end()) 
		{
		  edm::LogError("NotFound") << "pos,len not found for Ali in maps!";
		  return AlgebraicMatrix();
		}
	  ndimj += (*ilenmap).second;
	}


  // OK, let's do the real work now
  AlgebraicMatrix result(ndimi,ndimj,0);
  
  int iresi=1;
  for ( std::vector<Alignable*>::const_iterator it = veci.begin();
		it != veci.end(); ++it ) 
	{
	  Aliposmap::const_iterator iposmapi = theAliposmap.find( *it );
	  Alilenmap::const_iterator ilenmapi = theAlilenmap.find( *it );
	  int posi=(*iposmapi).second;
	  int leni=(*ilenmapi).second;
	  int iresj=1;
	  for ( std::vector<Alignable*>::const_iterator jt = vecj.begin();
			jt != vecj.end(); ++jt ) 
		{
		  Aliposmap::const_iterator iposmapj = theAliposmap.find( *jt );
		  Alilenmap::const_iterator ilenmapj = theAlilenmap.find( *jt );
		  int posj = (*iposmapj).second;
		  int lenj = (*ilenmapj).second;

		  AlgebraicMatrix piece(leni,lenj,0);
		  for (int ir=0;ir<piece.num_row();++ir)
			for (int ic=0;ic<piece.num_col();++ic)
			  piece[ir][ic] = theData->covariance()[posi+ir-1][posj+ic-1];
		  result.sub(iresi,iresj,piece);
		  iresj += lenj;
		}
	  iresi += leni;
	}
  
  return result;

}
