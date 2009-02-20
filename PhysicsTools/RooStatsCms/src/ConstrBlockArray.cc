
// @(#)root/hist:$Id: ConstrBlockArray.cc,v 1.1 2009/01/06 12:22:43 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008

#include "assert.h"

#include "PhysicsTools/RooStatsCms/interface/ConstrBlockArray.h"

/*----------------------------------------------------------------------------*/

ConstrBlockArray::ConstrBlockArray (const char* name, 
                                    const char* title,
                                    NLLPenalty* penalty1,
                                    NLLPenalty* penalty2,
                                    NLLPenalty* penalty3,
                                    NLLPenalty* penalty4,
                                    NLLPenalty* penalty5,
                                    NLLPenalty* penalty6,
                                    NLLPenalty* penalty7,
                                    NLLPenalty* penalty8):

    TNamed(name,title),
    m_size(0){
    setVerbosity(false);

    if (penalty1!=NULL)
        add(penalty1);
    if (penalty2!=NULL)
        add(penalty2);
    if (penalty3!=NULL)
        add(penalty3);
    if (penalty4!=NULL)
        add(penalty4);
    if (penalty5!=NULL)
        add(penalty5);
    if (penalty6!=NULL)
        add(penalty6);
    if (penalty7!=NULL)
        add(penalty7);
    if (penalty8!=NULL)
        add(penalty8);

    }

/*----------------------------------------------------------------------------*/

void ConstrBlockArray::add(NLLPenalty* penalty){
    m_penalties[m_size]=penalty;
    m_penalties[m_size]->setVerbosity(is_verbose());
    m_size++;
    std::cout << "m_size is now " << m_size << std::endl;
    }

/*----------------------------------------------------------------------------*/

void ConstrBlockArray::print(const char* options){
    std::cout << "ConstrBlockArray " << GetName()
              << " penalties are:\n";

    int penalty_index=1;
    NLLPenalty *pen;
    for(int i=0;i< m_size;++i){
        pen=m_penalties[i];
        std::cout << "\n* Penalty  " << penalty_index++ << " - " << pen << std::endl; 
        pen->print(options);
        }
    }

/*----------------------------------------------------------------------------*/

void ConstrBlockArray::restore(){
    NLLPenalty *pen;
    for(int i=0;i< m_size;++i){
        pen=m_penalties[i];
        pen->restore();
        }
    }

/*----------------------------------------------------------------------------*/

void ConstrBlockArray::fluctuate(){
    NLLPenalty *pen;
    for(int i=0;i< m_size;++i){
        pen=m_penalties[i];
        pen->fluctuate();
        }
    }

/*----------------------------------------------------------------------------*/

void ConstrBlockArray::setFixed(bool fix){
    NLLPenalty *pen;
    for(int i=0;i< m_size;++i){
        pen=m_penalties[i];
        pen->setFixed(fix);
        }
    }

/*----------------------------------------------------------------------------*/

TString ConstrBlockArray::getNLLstring(){
    TString nllstring="";

    NLLPenalty *pen;
    for(int i=0;i< m_size;++i){
        pen=m_penalties[i];
        nllstring+=pen->getNLLstring();
        }

    return nllstring;
    }

/*----------------------------------------------------------------------------*/

TString ConstrBlockArray::getBkgNLLstring(){
    TString bkgnllstring="";

    NLLPenalty *pen;
    for(int i=0;i< m_size;++i){
        pen=m_penalties[i];
        bkgnllstring+=pen->getBkgNLLstring();
        }

    return bkgnllstring;
    }

/*----------------------------------------------------------------------------*/

RooArgList ConstrBlockArray::getNLLterms(){

    RooArgList parlist;

    NLLPenalty *pen;
    for(int i=0;i< m_size;++i){
        if (is_verbose())
            std::cout << "Getting NLL terms for block: " << i << "\n";
        pen=m_penalties[i];
        parlist.add(pen->getNLLterms());
        }

    return parlist;

    }

/*----------------------------------------------------------------------------*/

RooArgList ConstrBlockArray::getBkgNLLterms(){

    RooArgList bkgparlist;

    NLLPenalty *pen;
    for(int i=0;i< m_size;++i){
        pen=m_penalties[i];
        bkgparlist.add(pen->getBkgNLLterms());
        }

    return bkgparlist;

    }

/*----------------------------------------------------------------------------*/

NLLPenalty* ConstrBlockArray::getBlock(int index){

    if (index>=m_size){
        NLLPenalty* p=0;
        return p;
        }
    else
        return m_penalties[index];
    }

/*----------------------------------------------------------------------------*/

ConstrBlockArray::~ConstrBlockArray(){
    // nothing to destruct
    }

/*----------------------------------------------------------------------------*/

/// To build the cint dictionaries
//ClassImp(ConstrBlockArray)

/*----------------------------------------------------------------------------*/
