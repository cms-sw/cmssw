// @(#)root/hist:$Id: RscConstrArrayFiller.cc,v 1.3 2009/04/15 11:10:44 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008

#include "assert.h"

#include <iostream>

#include "TIterator.h"

#include "PhysicsTools/RooStatsCms/interface/ConstrBlock2.h"
#include "PhysicsTools/RooStatsCms/interface/ConstrBlock3.h"


#include "PhysicsTools/RooStatsCms/interface/RscConstrArrayFiller.h"


/*----------------------------------------------------------------------------*/

RscConstrArrayFiller::RscConstrArrayFiller(char* name, 
                                           char* title, 
                                           RscCombinedModel* combo,
                                           bool verbosity)
    :RscTool(name,title,verbosity){

    if (is_verbose())
        std::cout << "[RscConstrArrayFiller::RscConstrArrayFiller] "
                  << "Combination size " << combo->getSize() << std::endl;

    for (int i=0;i<combo->getSize();++i)
        m_add_selective(combo->getModel(i)->getConstraints());

    if (is_verbose()){
        std::cout << "[RscConstrArrayFiller::RscConstrArrayFiller] "
                  << "The constraints: \n" <<  std::endl;
        m_constraints.Print();
        }

    }

/*----------------------------------------------------------------------------*/

RscConstrArrayFiller::RscConstrArrayFiller(char* name, 
                                           char* title, 
                                           RooArgList constraints_list,
                                           bool verbosity)
    :RscTool(name,title,verbosity){
    m_add_selective(&constraints_list);
    }

/*----------------------------------------------------------------------------*/

RscConstrArrayFiller::RscConstrArrayFiller(char* name,
                                            char* title,
                                            RscTotModel* model1,
                                            RscTotModel* model2,
                                            RscTotModel* model3,
                                            RscTotModel* model4,
                                            RscTotModel* model5,
                                            RscTotModel* model6,
                                            RscTotModel* model7,
                                            RscTotModel* model8,
                                            bool verbosity)
    :RscTool(name,title,verbosity){

    m_add_selective(model1->getConstraints());

    if (model2!=0)
        m_add_selective(model2->getConstraints());
    if (model3!=0)
        m_add_selective(model3->getConstraints());
    if (model4!=0)
        m_add_selective(model4->getConstraints());
    if (model5!=0)
        m_add_selective(model5->getConstraints());
    if (model6!=0)
        m_add_selective(model6->getConstraints());
    if (model7!=0)
        m_add_selective(model7->getConstraints());
    if (model8!=0)
        m_add_selective(model8->getConstraints());

    if (is_verbose()){
        std::cout << "[RscConstrArrayFiller::RscConstrArrayFiller] "
                  << "The constraints: \n" <<  std::endl;
        m_constraints.Print();
        }
    }

/*----------------------------------------------------------------------------*/

void RscConstrArrayFiller::m_add_selective(RooArgList* list){

    TString constraint_ClassName=("Constraint");

    for (int i=0;i<list->getSize();++i)
        if (not m_constraints.contains((*list)[i])){
            if ((*list)[i].ClassName()!=constraint_ClassName){
                std::cout << "Parameter " << (*list)[i].GetName()
                          << " is not a Constraint ... Aborting!!\n";
                abort();
                }
            m_constraints.add((*list)[i]);
            }
    }

/*----------------------------------------------------------------------------*/

Constraint* RscConstrArrayFiller::m_pop(TString name){

    if (is_verbose())
        std::cout << "[RscConstrArrayFiller::m_pop] Entering \n";

    Constraint* myconstr;

    for (int i=0;i<m_constraints.getSize();++i){
        if ((m_constraints[i]).GetName()==name){
            myconstr=dynamic_cast<Constraint*> (&(m_constraints[i]));
            if (is_verbose())
                std::cout << "[RscConstrArrayFiller::m_pop] "
                          << " popping variable " << name << "...\n";
            //m_constraints.remove(*myconstr);
            m_corr_constraints.add(*myconstr);
            return myconstr;
            }
        }

    std::cout << "ERROR : [RscConstrArrayFiller::m_pop] "
              << "constraint " << name << " not found... Aborting!!\n";
    abort();
    }

/*----------------------------------------------------------------------------*/

/**
Inspect the datacard to find the correlation blocks, marked by the 
[constraints_blok_N] title, where N is the number fo the block.
The blocks are seeked for in order, starting from 1.
Once the block is found the method m_fill_single_block is found.
**/

void RscConstrArrayFiller::fill (ConstrBlockArray* the_array, const char* blocknamebase){

    TString block_name_base=blocknamebase;
    block_name_base+="_constraints_block_";
    TString block_name="";
    NLLPenalty* block;

    int block_index=1;
    while (true){
        block_name=block_name_base;
        block_name+=block_index;
        block=m_fill_single_block(block_name);

        if (block==NULL)
            break;

        the_array->add(block);

        block_index++;
        }

    // and now the uncorrelated params:
    for (int i=0;i< m_constraints.getSize();++i){
        if (not m_corr_constraints.contains(m_constraints[i])){
            the_array->add( dynamic_cast<NLLPenalty*> (&(m_constraints[i])));
            if (is_verbose())
                std::cout << "[RscConstrArrayFiller::fill] "
                        << "Filling with uncorrelated constraint " 
                        << m_constraints[i].GetName() << std::endl;
            }
        }

    }

/*----------------------------------------------------------------------------*/

/**
Read the correlations in a single block and fills a CorrelationsBlock2 or
CorrelationBlock3 according to their number. The pointer to the block is then 
returned.
**/

NLLPenalty* RscConstrArrayFiller::m_fill_single_block (TString block_name){

    int number_of_constraints=0;

    TString variable_label;

    TString variables_names[4];
    double correlation_values[3];

    bool vars_ended=false;

    std::cout << "Block name: " << block_name.Data() << std::endl;

    // loop on the constraints names.
    while(not vars_ended){

        variable_label="correlation_variable";
        variable_label+=number_of_constraints+1;

        RooStringVar var_name(variable_label.Data(),"","");

        //std::cout << "Seeking for variable: " << var_name.GetName() << "\n";

        RooArgSet(var_name).readFromFile(RscAbsPdfBuilder::getDataCard(),
                                         0,
                                         block_name.Data());


        variables_names[number_of_constraints]=var_name.getVal();

        // If the string it's not there quit the search
        if (variables_names[number_of_constraints]==""){
            vars_ended=true;
            continue;
            }


        if (is_verbose())
            std::cout << "[RscConstrArrayFiller::m_fill_single_block] " 
                      << " constraint name in the block: " << var_name.getVal()
                      << std::endl;


        number_of_constraints++;
        }

    //std::cout << "NUMBER OF CONSTRAINTS: " << number_of_constraints << "\n";


    // No constraints in the block. We assume that the block does not exist
    if (number_of_constraints==0)
        return 0;

    // either 2 or 3 correlated vars
    assert (number_of_constraints==2 or number_of_constraints==3);

    // loop for the correlation values. By hand their number.
    int number_of_correlations=0;
    if (number_of_constraints==3)
        number_of_correlations=3;
    if (number_of_constraints==2)
        number_of_correlations=1;

    for (int corr_index=0;corr_index<number_of_correlations;++corr_index){

        variable_label="correlation_value";
        variable_label+=(corr_index+1);

        std::cout << "Seeking " << variable_label.Data() << std::endl;

        RooRealVar corr(variable_label.Data(),"",-1);
        RooArgSet(corr).readFromFile(RscAbsPdfBuilder::getDataCard(),
                                     0,
                                     block_name.Data());


        if (is_verbose())
            std::cout << "[RscConstrArrayFiller::m_fill_single_block] " 
                      << " constraint value in the block: " << corr.getVal()
                      << std::endl;


        correlation_values[corr_index]=corr.getVal();

        assert(corr.getVal()<1 and corr.getVal()>0);

        // FIXME to avoid errors in the matrix inversion!
        if (correlation_values[number_of_constraints-1]==1)
            correlation_values[number_of_constraints-1]=0.99;

        }

    // Fill the correlation block
    NLLPenalty* block=0http://www.physik.uni-karlsruhe.de/Personen.php?id=13;
    // Fill the correlation block
    if (number_of_constraints==2){
        if (is_verbose())
            std::cout << "[RscConstrArrayFiller::m_fill_single_block] "
                         " filling a ConstrBlock2 ...\n";

        block = new ConstrBlock2 (const_cast<char*> (block_name.Data()),
                                  const_cast<char*> (block_name.Data()),
                                  correlation_values[0],
                                  m_pop(variables_names[0]),
                                  m_pop(variables_names[1]));

        }

    if (number_of_constraints==3){
        if (is_verbose())
            std::cout << "[RscConstrArrayFiller::m_fill_single_block] "
                         " filling a ConstrBlock3 ...\n";

        block = new ConstrBlock3 (const_cast<char*> (block_name.Data()),
                                  const_cast<char*> (block_name.Data()),
                                  correlation_values[0],
                                  correlation_values[1],
                                  correlation_values[2],
                                  m_pop(variables_names[0]),
                                  m_pop(variables_names[1]),
                                  m_pop(variables_names[2]));
        }

    return block;

    }

/*----------------------------------------------------------------------------*/


// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
