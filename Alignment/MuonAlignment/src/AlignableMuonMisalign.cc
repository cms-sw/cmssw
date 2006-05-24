#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/Randomize.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/MuonAlignment/interface/AlignableMuonMisalign.h"


//__________________________________________________________________________________________________
AlignableMuonMisalign::AlignableMuonMisalign()
{

  edm::LogInfo("BeginModify") 
	<< " ========> Instantiating a modified muon   <==========" << std::endl;
  
}


//__________________________________________________________________________________________________
AlignableMuonMisalign::~AlignableMuonMisalign()
{

  edm::LogInfo("EndModify")
	<< " ========> Finishing the muon components modification <========" << std::endl;
  
}


//__________________________________________________________________________________________________
/// Random gaussian move. If setSeed is true, the seed is taken from the last argument.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableMuonMisalign::randomMove( std::vector<Alignable*> comp, 
					float sigmaX, float sigmaY,float sigmaZ, 
					bool setSeed, long seed )
{

  edm::LogInfo("PrintArgs") << "move  randomly  with sigma " 
							<< sigmaX << " " << sigmaY << " " << sigmaZ 
							<< std::endl;
  
  DRand48Engine aDRand48Engine;
  if (setSeed) 
	aDRand48Engine.setSeed( seed, 0 );
  else
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  aDRand48Engine.setSeed( rng->mySeed(), 0 );
	}

  RandGauss aGaussObjX( aDRand48Engine, 0., sigmaX );
  RandGauss aGaussObjY( aDRand48Engine, 0., sigmaY );
  RandGauss aGaussObjZ( aDRand48Engine, 0., sigmaZ );

  for ( std::vector<Alignable*>::iterator i=comp.begin(); i<comp.end(); i++ )
	{
	  GlobalVector moveV( aGaussObjX.fire(), aGaussObjY.fire(),
						  aGaussObjZ.fire() );
	  (*i)->move(moveV);
	}

}


//__________________________________________________________________________________________________
/// Random flat move. If setSeed is true, the seed is taken from the last argument.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableMuonMisalign::randomFlatMove( std::vector<Alignable*> comp, 
					    float sigmaX, float sigmaY, float sigmaZ,
					    bool setSeed, long seed )
{

  edm::LogInfo("PrintArgs") << "move  randomly  with sigma " 
							<< sigmaX << " " << sigmaY << " " << sigmaZ 
							<< std::endl;
  
  DRand48Engine aDRand48Engine;
  if (setSeed)
	aDRand48Engine.setSeed(seed, 0);
  else
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  aDRand48Engine.setSeed( rng->mySeed(), 0 );
	}
  
  RandFlat aFlatObjX( aDRand48Engine, sigmaX );
  RandFlat aFlatObjY( aDRand48Engine, sigmaY );
  RandFlat aFlatObjZ( aDRand48Engine, sigmaZ );

  for ( std::vector<Alignable*>::iterator i=comp.begin(); i<comp.end(); i++ )
	{
	  GlobalVector moveV( aFlatObjX.fire(), aFlatObjY.fire(),
						  aFlatObjZ.fire() );
	  (*i)->move(moveV);
	}

}


//__________________________________________________________________________________________________
/// Random gaussian movement of all components of the collection of Alignables
/// (which have to be AlignableComposites) within the super structure. 
/// X,Y,Z axis are interpreted as local coordinates in the Composite. 
/// WATCHOUT. If the vector you supply here is a vector of rod/petals, 
/// the GeomDets on them will be moved "locally".
/// If setSeed is true, the seed is taken from the last argument.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableMuonMisalign::randomMoveComponentsLocal( std::vector<Alignable*> comp,
						       float sigmaX, float sigmaY,float sigmaZ, 
						       bool setSeed, long seed )
{

  edm::LogInfo("PrintArgs") << "move  randomly  local within the composite structure with sigma " 
							<< sigmaX << " " << sigmaY << " " << sigmaZ 
							<< std::endl;
  
  DRand48Engine aDRand48Engine;
  if (setSeed)
	aDRand48Engine.setSeed(seed, 0);
  else
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  aDRand48Engine.setSeed( rng->mySeed(), 0 );
	}
  
  RandGauss aGaussObjX( aDRand48Engine, 0., sigmaX );
  RandGauss aGaussObjY( aDRand48Engine, 0., sigmaY );
  RandGauss aGaussObjZ( aDRand48Engine, 0., sigmaZ );

  for ( std::vector<Alignable*>::iterator i=comp.begin(); i<comp.end(); i++ )
	{
	  if ( (*i)->size() ) 
		{
		  LocalVector moveV( aGaussObjX.fire(), aGaussObjY.fire(), aGaussObjZ.fire() );
		  (static_cast<AlignableComposite*>(*i))->moveComponentsLocal(moveV);
		}
	  else 
		edm::LogError("LogicError") 
		  <<"Cannot rotate components: size of the Composite is zero." << std::endl;
	}

}


//__________________________________________________________________________________________________
/// Random gaussian rotation. If setSeed is true, the seed is taken from the last argument.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableMuonMisalign::randomRotate( std::vector<Alignable*> comp, 
					  float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ, 
					  bool setSeed, long seed )
{
  edm::LogInfo("PrintArgs") << "rotate  randomly  about GLOBAL x,y,z axis with sigmaPhi " 
							<< sigmaPhiX << " " << sigmaPhiY << " " << sigmaPhiZ 
							<< std::endl;
  
  DRand48Engine aDRand48Engine;
  if (setSeed) aDRand48Engine.setSeed(seed, 0);
  else
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  aDRand48Engine.setSeed( rng->mySeed(), 0 );
	}

  RandGauss aGaussObjX( aDRand48Engine, 0., sigmaPhiX );
  RandGauss aGaussObjY( aDRand48Engine, 0., sigmaPhiY );
  RandGauss aGaussObjZ( aDRand48Engine, 0., sigmaPhiZ ); 
  
  for ( std::vector<Alignable*>::iterator i=comp.begin(); i<comp.end(); i++ )
	{
	  (*i)->rotateAroundGlobalX(aGaussObjX.fire());
	  (*i)->rotateAroundGlobalY(aGaussObjY.fire());
	  (*i)->rotateAroundGlobalZ(aGaussObjZ.fire());
	}

}


//__________________________________________________________________________________________________
/// Here the rotation Axis is interpreted according to the local coordinate system of the Alignable.
/// First it is rotated around local_x, then the new local_y and then the new local_z.
/// If setSeed is true, the seed is taken from the last argument.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableMuonMisalign::randomRotateLocal( std::vector<Alignable*> comp, 
					       float sigmaPhiX, float sigmaPhiY,float sigmaPhiZ, 
					       bool setSeed, long seed )
{

  edm::LogInfo("PrintArgs") << "rotate  randomly  around LOCAL x,y,z axis with sigmaPhi " 
							<< sigmaPhiX << " " << sigmaPhiY << " " << sigmaPhiZ 
							<< std::endl;
  
  DRand48Engine aDRand48Engine;
  if (setSeed) aDRand48Engine.setSeed(seed, 0);
  else
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  aDRand48Engine.setSeed( rng->mySeed(), 0 );
	}

  RandGauss aGaussObjX( aDRand48Engine, 0., sigmaPhiX );
  RandGauss aGaussObjY( aDRand48Engine, 0., sigmaPhiY );
  RandGauss aGaussObjZ( aDRand48Engine, 0., sigmaPhiZ );

  for ( std::vector<Alignable*>::iterator i=comp.begin(); i<comp.end(); i++ )
	{
	  (*i)->rotateAroundLocalX(aGaussObjX.fire());
	  (*i)->rotateAroundLocalY(aGaussObjY.fire());
	  (*i)->rotateAroundLocalZ(aGaussObjZ.fire());
	}

}


//__________________________________________________________________________________________________
/// Here the rotation Axis is interpreted according to the local coordinate system of the Alignable.
/// First it is rotated around local_x, then the new local_y and then the new local_z.
/// If setSeed is true, the seed is taken from the last argument.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableMuonMisalign::randomFlatRotateLocal( std::vector<Alignable*> comp, 
						   float sigmaPhiX, float sigmaPhiY,float sigmaPhiZ, 
						   bool setSeed, long seed )
{

  edm::LogInfo("PrintArgs") << "Rotate  randomly  around LOCAL x,y,z axis with sigmaPhi " 
							<< sigmaPhiX << " " << sigmaPhiY << " " << sigmaPhiZ 
							<< std::endl;
  
  DRand48Engine aDRand48Engine;
  if (setSeed) aDRand48Engine.setSeed(seed, 0);
  else
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  aDRand48Engine.setSeed( rng->mySeed(), 0 );
	}


  RandFlat aFlatObjX( aDRand48Engine, sigmaPhiX );
  RandFlat aFlatObjY( aDRand48Engine, sigmaPhiY );
  RandFlat aFlatObjZ( aDRand48Engine, sigmaPhiZ );
  
  for ( std::vector<Alignable*>::iterator i=comp.begin(); i<comp.end(); i++ )
	{
	  (*i)->rotateAroundLocalX(aFlatObjX.fire());
	  (*i)->rotateAroundLocalY(aFlatObjY.fire());
	  (*i)->rotateAroundLocalZ(aFlatObjZ.fire());
	}

}


//__________________________________________________________________________________________________
/// Random gaussian rotation of all components of the collection of Alignables
/// (which should be AlignableComposites) within the super structure. 
/// X,Y,Z axis are interpreted as local coordinates in the Composite. 
/// WATCHOUT. If the vector you supply here is a vector of rod/petals, 
/// the GeomDets on them will be moved "locally".
/// If setSeed is true, the seed is taken from the last argument.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableMuonMisalign::randomRotateComponentsLocal( std::vector<Alignable*> comp,
							 float sigmaPhiX, float sigmaPhiY,
							 float sigmaPhiZ,
							 bool setSeed, long seed )
{

  edm::LogInfo("PrintArgs") << "Rotate  randomly  local within the composite structure "
							<< std::endl << "with sigma and its local x,y,z axis with sigmaPhi: "
							<< sigmaPhiX << " " << sigmaPhiY << " " << sigmaPhiZ << std::endl;
  
  DRand48Engine aDRand48Engine;
  if (setSeed) aDRand48Engine.setSeed(seed, 0); 
  else
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  aDRand48Engine.setSeed( rng->mySeed(), 0 );
	}

  RandGauss aGaussObjX( aDRand48Engine, 0., sigmaPhiX );
  RandGauss aGaussObjY( aDRand48Engine, 0., sigmaPhiY );
  RandGauss aGaussObjZ( aDRand48Engine, 0., sigmaPhiZ );

  //simply loop over all components of each of the Alignables in the vector
  //and make the corresponding LOCAL rotation... (means using the coordinate
  //system (orientation) of the component itself
  for ( std::vector<Alignable*>::iterator i=comp.begin(); i<comp.end(); i++ )
	{
	  if ( (*i)->size() ) 
		{
		  std::vector<Alignable*> comp2 = ((AlignableComposite*)(*i))->components();
		  for (std::vector<Alignable*>::iterator k=comp2.begin();k<comp2.end();k++){
			(*k)->rotateAroundLocalX(aGaussObjX.fire());
			(*k)->rotateAroundLocalY(aGaussObjY.fire());
			(*k)->rotateAroundLocalZ(aGaussObjZ.fire());
		  }
		} 
	  else 
		edm::LogError("LogicError") 
		  <<"Cannot rotate components: size of the Composite is zero." << std::endl;
	}

}



//__________________________________________________________________________________________________
void AlignableMuonMisalign::addAlignmentPositionError( std::vector<Alignable*> comp, 
						       float dx, float dy, float dz )
{

  edm::LogInfo("PrintArgs") << "Adding an AlignmentPositionError of size " 
							<< dx << " "  << dy << " "  << dz << std::endl;

  for ( std::vector<Alignable*>::iterator i=comp.begin(); i<comp.end(); i++ )
	{
	  AlignmentPositionError ape(dx,dy,dz);
	  (*i)->addAlignmentPositionError(ape);
	}  
}


//__________________________________________________________________________________________________
void AlignableMuonMisalign::addAlignmentPositionErrorLocal( std::vector<Alignable*> comp, 
							    float dx, float dy, float dz )
{

  // Transform the given dx,dy,dz (interpreted as local coordinates within
  // the corresponding Alignable* (it should be a composite) and then, the
  // corresponding APE is added to the components. As the 
  // AlignmentPositionError  of a composite is automatically forwarded to its
  // components, it is enough to just call the addAPE method for each
  // composite.

  LocalVector v(dx,dy,dz);
  edm::LogInfo("PrintArgs") << "Adding an AlignmentPositionError of local size " << v << std::endl; 

  for ( std::vector<Alignable*>::iterator i=comp.begin(); i<comp.end(); i++ )
	{
	  const GlobalVector gv = (*i)->surface().toGlobal(v);
	  GlobalVector::BasicVectorType bv=gv.basicVector();
	  AlignmentPositionError ape( bv.x(), bv.y(), bv.z() );
	  (*i)->addAlignmentPositionError(ape);
	}  
}


//__________________________________________________________________________________________________
void AlignableMuonMisalign::addAlignmentPositionErrorFromRotation( std::vector<Alignable*> comp, 
								   RotationType& rotation )
{ 

  edm::LogInfo("PrintArgs") << " adding an AlignmentPositionError from Rotation" << std::endl 
							<< rotation << std::endl;

  for ( std::vector<Alignable*>::iterator i=comp.begin(); i<comp.end(); i++ )
	(*i)->addAlignmentPositionErrorFromRotation(rotation);

}


//__________________________________________________________________________________________________
void AlignableMuonMisalign::addAlignmentPositionErrorFromLocalRotation( std::vector<Alignable*> comp, 
								        RotationType& rotation )
{ 
  
  edm::LogInfo("PrintArgs") << " adding an AlignmentPositionError from Local Rotation" << std::endl 
							<< rotation << std::endl;
  
  for ( std::vector<Alignable*>::iterator i=comp.begin(); i<comp.end(); i++ )
	(*i)->addAlignmentPositionErrorFromLocalRotation(rotation);
  
}

