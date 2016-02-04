#include <cassert>
#include <iostream>
#include <iomanip>
using namespace std;

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/ME.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEChannel.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEGeom.h"

// ClassImp(MEChannel)

MEChannel::MEChannel( int ix, int iy, int id_, MEChannel* mother )
  : _m(mother)
{
  if( _m==0 ) 
    {
      _ig=0;      
    }
  else 
    {
      _ig = _m->_ig+1;
    }
  _id.resize( _ig+1, -1 );
  for( int ii=0; ii<_ig; ii++ )
    {
      _id[ii] = _m->_id[ii];
    }
  _id[_ig] = id_;
  _ix = ix;
  _iy = iy;
}

MEChannel::~MEChannel() 
{
  for( unsigned ii=0; ii<_d.size(); ii++ )
    {
      delete _d[ii];
    }
}

MEChannel* 
MEChannel::getDaughter( int ix, int iy, int id_ )
{
  for( unsigned ii=0; ii<_d.size(); ii++ )
    {
      if( _d[ii]->id()==id_ ) 
	{
	  return _d[ii];
	}
    }  
  return addDaughter( ix, iy, id_ );
}

MEChannel*
MEChannel::addDaughter( int ix, int iy, int id_ )
{
  MEChannel* d = new MEChannel( ix, iy, id_, this );
  _d.push_back( d );
  return d;
}

int
MEChannel::id() const
{
  return _id[_ig];
}

bool
MEChannel::getListOfChannels( std::vector< MEChannel* >& vec )
{
  if( n()==0 ) 
    {
      vec.push_back( this );
      return true;
    }
  for( unsigned ii=0; ii<n(); ii++ )
    {
      bool ok = _d[ii]->getListOfChannels( vec ); 
      assert( ok );
    }
  return true;
}

bool
MEChannel::getListOfAncestors( std::vector< MEChannel* >& vec )
{
  MEChannel* mother = this->m();
  if( mother!=0 )
    {
      vec.push_back( mother );
      mother->getListOfAncestors( vec );
    }
  return true;
}

MEChannel* 
MEChannel::getAncestor( int g )
{
  if( _ig==g ) return this;
  
  MEChannel* mother = this->m();
  if( mother!=0 )
    {
      if( mother->_ig==g ) return mother;
      return mother->getAncestor( g );
    }

  return 0;
}

bool
MEChannel::getListOfDescendants( std::vector< MEChannel* >& vec )
{
  for( unsigned ii=0; ii<n(); ii++ )
    {
      vec.push_back(_d[ii]);
      _d[ii]->getListOfDescendants( vec ); 
    }
  return true;
}

bool
MEChannel::getListOfDescendants( int ig, std::vector< MEChannel* >& vec )
{
  for( unsigned ii=0; ii<n(); ii++ )
    {
      MEChannel* curLeaf = _d[ii];
      if( curLeaf->_ig==ig ) vec.push_back(curLeaf);
      curLeaf->getListOfDescendants( ig, vec ); 
    }
  return true;
}

MEChannel*
MEChannel::getDescendant( int ig, int id_ )
{
  std::vector< MEChannel* > vec;
  bool OK = getListOfDescendants( ig, vec );
  if( !OK ) return 0;
  MEChannel* leaf(0);
  for( unsigned int ii=0; ii<vec.size(); ii++ )
    {
      leaf = vec[ii];
      if( leaf->id()==id_ ) return leaf;
    }
  return leaf;
}

MEChannel*
MEChannel::getFirstDescendant( int ig )
{
  std::vector< MEChannel* > vec;
  bool OK = getListOfDescendants( ig, vec );
  if( !OK ) return 0;
  return vec[0];
}

MEChannel* 
MEChannel::getChannel( int ig, int ix, int iy )
{
  assert( ig>=0 );
  MEChannel* leaf = getChannel( ix, iy );
  if( leaf==0 ) return 0;
  while( ig!=leaf->_ig )
    {
      leaf = leaf->_m;
    }
  return leaf;
}

MEChannel*
MEChannel::getChannel( int ix, int iy )
{
  if( n()==0 )
    {
      if( ix==_ix && iy==_iy )
	{
	  return this;
	}
      else
	return 0;
    }
  MEChannel* leaf(0);
  for( unsigned ii=0; ii<n(); ii++ )
    {
      leaf =  _d[ii]->getChannel( ix, iy ); 
      if( leaf!=0 ) break;
    }
  return leaf;
}

void
MEChannel::print( ostream& o, bool recursif ) const
{
  o << ME::granularity[_ig] << " ";
  for( int ii=0; ii<=_ig; ii++ )
    { 
      o << ME::granularity[ii] << "=" << _id[ii] << " ";
    }
  if( n()>0 )
    {
      o << "NDau=" << n() << " " ;
    }
  else
    {
      o << "ix=" << _ix << " iy=" << _iy << " " ;
    }
  o << std::endl;
  if( recursif )
    {
      for( unsigned jj=0; jj<n(); jj++ )
	{
	  _d[jj]->print( o, true );
	}
    }
}

TString
MEChannel::oneLine( int ig )
{
  assert( ig>=ME::iEcalRegion );
  int reg_ = _id[ME::iEcalRegion];
  TString out;
  if( ig<ME::iLMRegion )
    {
      out += ME::region[reg_];
      if( ig==ME::iSector ) out+="/S="; out+=_id[ME::iSector];
      return out;
    }
  int lmr_ = _id[ME::iLMRegion];
  std::pair<int,int> p_ = ME::dccAndSide( lmr_ );
  out+= ME::granularity[ME::iLMRegion]; 
  out+="=";out += lmr_;
  int dcc_=p_.first; 
  int side_=p_.second; 
  out+="(DCC="; out+=dcc_; out+=","; 
  out+= ME::smName(lmr_);
  out+="/"; out+=side_; out+=")";
  if( ig>=_ig ) ig=_ig;
  if( ig>=ME::iLMModule )
    {
      int lmm_=_id[ME::iLMModule]; 
      out+="/"; 
      out+=ME::granularity[ME::iLMModule];
      out+="=";
      out+=lmm_;
      if( ig>=ME::iSuperCrystal )
	{
	  int sc_=_id[ME::iSuperCrystal];
	  out+="/"; 
	  out+=ME::granularity[ME::iSuperCrystal];
	  out+="=";
	  out+=sc_;
	  if( ig>=ME::iCrystal )
	    {
	      int c_=_id[ME::iCrystal];
	      out+="/"; 
	      out+=ME::granularity[ME::iCrystal];
	      out+="=";
	      out+=c_;
	      if( reg_==ME::iEBM || reg_==ME::iEBP )
		{
		  out += "/ieta="; out+=ix();
		  out += "/iphi="; out+=iy();
		  MEEBGeom::XYCoord ixy_ = 
		    MEEBGeom::localCoord( ix(), iy() );
		  out += "(ix="; out+=ixy_.first;
		  out += "/iy="; out+=ixy_.second;
		  out += ")";
		}
	      else
		{
		  out += "/ix="; out+=ix();
		  out += "/iy="; out+=iy();
		}
	    }
	}
    }
  if( ig<ME::iCrystal )
    {
      std::vector< MEChannel* > _channels;
      getListOfChannels( _channels );
      int nchan = _channels.size();
      if( nchan>1 )
	{
	  out += "(";
	  out += nchan;
	  out += "xTals)";
	}
    }
  return out;
}

TString
MEChannel::oneWord( int ig )
{
  assert( ig>=ME::iEcalRegion );
  int reg_ = _id[ME::iEcalRegion];
  TString out;
  if( ig<ME::iLMRegion )
    {
      out = "ECAL_"; 
      out += ME::region[reg_];
      if( ig==ME::iSector ) out+="_S"; out+=_id[ME::iSector];
      return out;
    }
  int lmr_ = _id[ME::iLMRegion];
  out+= ME::granularity[ME::iLMRegion]; 
  out+=lmr_;
  if( ig>=_ig ) ig=_ig;
  if( ig>=ME::iLMModule )
    {
      int lmm_=_id[ME::iLMModule]; 
      out+="_"; 
      out+=ME::granularity[ME::iLMModule];
      out+=lmm_;
      if( ig>=ME::iSuperCrystal )
	{
	  int sc_=_id[ME::iSuperCrystal];
	  out+="_"; 
	  out+=ME::granularity[ME::iSuperCrystal];
	  out+=sc_;
	  if( ig>=ME::iCrystal )
	    {
	      int c_=_id[ME::iCrystal];
	      out+="_"; 
	      out+=ME::granularity[ME::iCrystal];
	      out+=c_;
	      if( reg_==ME::iEBM || reg_==ME::iEBP )
		{
		  MEEBGeom::XYCoord ixy_ = 
		    MEEBGeom::localCoord( ix(), iy() );
		  out += "_"; out+=ixy_.first;
		  out += "_"; out+=ixy_.second;
		}
	      else
		{
		  out += "_"; out+=ix();
		  out += "_"; out+=iy();
		}
	    }
	}
    }
  return out;
}

