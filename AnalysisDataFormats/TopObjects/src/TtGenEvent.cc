#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

TtGenEvent::TtGenEvent()
{
}

TtGenEvent::TtGenEvent(reco::CandidateRefProd& ref)
{
  parts_=ref;
}

TtGenEvent::~TtGenEvent()
{
}

int
TtGenEvent::numberOfLeptons() const
{
  int lep=0;
  reco::CandidateCollection::const_iterator part = parts_->begin();
  for( ; part!=parts_->end(); ++part ){
    if( isLepton(*part) ){
      ++lep;
    }
  }  
  return lep;  
}

const reco::Candidate*
TtGenEvent::candidate(int id) const
{
  const reco::Candidate* cand=0;
  reco::CandidateCollection::const_iterator part = parts_->begin();
  for( ; part!=parts_->end(); ++part ){
    if( part->pdgId()==id ) cand=&(*part);
  }  
  return cand;
}

const reco::Candidate* 
TtGenEvent::singleLepton() const 
{
  const reco::Candidate* cand=0;
  if( numberOfLeptons()==1 ){
    reco::CandidateCollection::const_iterator part = parts_->begin();
    for( ; part!=parts_->end(); ++part ){
      if( isLepton(*part) ){
	cand=&(*part);
      }  
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::singleNeutrino() const 
{
  const reco::Candidate* cand=0;
  if( numberOfLeptons()==1 ){
    reco::CandidateCollection::const_iterator part = parts_->begin();
    for( ; part!=parts_->end(); ++part ){
      if( isNeutrino(*part) ){
	cand=&(*part);
      }  
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::hadronicQuark() const 
{
  const reco::Candidate* cand=0;
  if( singleLepton() ){
    reco::CandidateCollection::const_iterator part = parts_->begin();
    for( ; part!=parts_->end(); ++part ){
      if( abs(part->pdgId()) < 5 && flavour(*part)>0 )
	cand=&(*part);
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::hadronicQuarkBar() const 
{
  const reco::Candidate* cand=0;
  if( singleLepton() ){
    reco::CandidateCollection::const_iterator part = parts_->begin();
    for( ; part!=parts_->end(); ++part ){
      if( abs(part->pdgId()) < 5 && flavour(*part)<0 )
	cand=&(*part);
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::hadronicB() const 
{
  const reco::Candidate* cand=0;
  if( singleLepton() ){
    reco::CandidateCollection::const_iterator part = parts_->begin();
    for( ; part!=parts_->end(); ++part ){
      if( abs(part->pdgId())==5 && 
	  flavour(*singleLepton())==flavour(*part) )
	cand=&(*part);
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::hadronicW() const 
{
  const reco::Candidate* cand=0;
  if( singleLepton() ){
    reco::CandidateCollection::const_iterator part = parts_->begin();
    for( ; part!=parts_->end(); ++part ){
      if( abs(part->pdgId())==24 && 
	  flavour(*singleLepton())!=flavour(*part) )
	cand=&(*part);
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::hadronicTop() const 
{
  const reco::Candidate* cand=0;
  if( singleLepton() ){
    reco::CandidateCollection::const_iterator part = parts_->begin();
    for( ; part!=parts_->end(); ++part ){
      if( abs(part->pdgId())==6 && 
	  flavour(*singleLepton())==flavour(*part) )
	cand=&(*part);
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::leptonicB() const 
{
  const reco::Candidate* cand=0;
  if( singleLepton() ){
    reco::CandidateCollection::const_iterator part = parts_->begin();
    for( ; part!=parts_->end(); ++part ){
      if( abs(part->pdgId())==5 && 
	  flavour(*singleLepton())!=flavour(*part) )
	cand=&(*part);
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::leptonicW() const 
{
  const reco::Candidate* cand=0;
  if( singleLepton() ){
    reco::CandidateCollection::const_iterator part = parts_->begin();
    for( ; part!=parts_->end(); ++part ){
      if( abs(part->pdgId())==24 && 
	  flavour(*singleLepton())==flavour(*part) )
	cand=&(*part);
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::leptonicTop() const 
{
  const reco::Candidate* cand=0;
  if( singleLepton() ){
    reco::CandidateCollection::const_iterator part = parts_->begin();
    for( ; part!=parts_->end(); ++part ){
      if( abs(part->pdgId())==6 && 
	  flavour(*singleLepton())!=flavour(*part) )
	cand=&(*part);
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::lepton() const 
{
  const reco::Candidate* cand=0;
  reco::CandidateCollection::const_iterator part = parts_->begin();
  for( ; part!=parts_->end(); ++part ){
    if( isLepton(*part) && flavour(*part)>0 ){
      cand=&(*part);
    }  
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::neutrino() const 
{
  const reco::Candidate* cand=0;
  reco::CandidateCollection::const_iterator part = parts_->begin();
  for( ; part!=parts_->end(); ++part ){
    if( isNeutrino(*part) && flavour(*part)>0 ){
      cand=&(*part);
    }  
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::leptonBar() const 
{
  const reco::Candidate* cand=0;
  reco::CandidateCollection::const_iterator part = parts_->begin();
  for( ; part!=parts_->end(); ++part ){
    if( isLepton(*part) && flavour(*part)<0 ){
      cand=&(*part);
    }  
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::neutrinoBar() const 
{
  const reco::Candidate* cand=0;
  reco::CandidateCollection::const_iterator part = parts_->begin();
  for( ; part!=parts_->end(); ++part ){
    if( isNeutrino(*part) && flavour(*part)<0 ){
      cand=&(*part);
    }  
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::quarkFromTop() const 
{
  const reco::Candidate* cand=0;
  reco::CandidateCollection::const_iterator part = parts_->begin();
  for( ; part!=parts_->end(); ++part ){
    if( part->mother() && flavour(*(part->mother()))<0
	&& abs(part->pdgId())<5 && flavour(*part)>0){
      cand=&(*part);
    }  
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::quarkFromTopBar() const 
{
  const reco::Candidate* cand=0;
  reco::CandidateCollection::const_iterator part = parts_->begin();
  for( ; part!=parts_->end(); ++part ){
    if( part->mother() && flavour(*(part->mother()))<0
	&& abs(part->pdgId())<5 && flavour(*part)<0){
      cand=&(*part);
    }  
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::quarkFromAntiTop() const 
{
  const reco::Candidate* cand=0;
  reco::CandidateCollection::const_iterator part = parts_->begin();
  for( ; part!=parts_->end(); ++part ){
    if( part->mother() && flavour(*(part->mother()))>0
	&& abs(part->pdgId())<5 && flavour(*part)>0){
      cand=&(*part);
    }  
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::quarkFromAntiTopBar() const 
{
  const reco::Candidate* cand=0;
  reco::CandidateCollection::const_iterator part = parts_->begin();
  for( ; part!=parts_->end(); ++part ){
    if( part->mother() && flavour(*(part->mother()))>0
	&& abs(part->pdgId())<5 && flavour(*part)<0){
      cand=&(*part);
    }  
  }
  return cand;
}
