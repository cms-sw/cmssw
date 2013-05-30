/*
 * Combinatorics.h
 *
 * 03/04/2006 kasselmann@physik.rwth-aachen.de
 *
 */

#ifndef COMBINATORICS_H
#define COMBINATORICS_H

// C++ header
#include <vector>

// ROOT header
#include <TROOT.h>

class Combinatorics
{
    
public:
    
    Combinatorics(Int_t Set, Int_t Subset);
    virtual ~Combinatorics();
    
    std::vector < std::vector <UInt_t> > GetPermutations();
    std::vector < std::vector <UInt_t> > GetCombinations();
    std::vector < std::vector <UInt_t> > GetCombinations_2_0();
    std::vector < std::vector <UInt_t> > GetCombinations_2_2();
    std::vector < std::vector <UInt_t> > GetCombinations_N_1();

    Int_t EqualPermutation(const std::vector<UInt_t>& permutation1, 
                           const std::vector<UInt_t>& permutation2);
    Int_t EqualPermutation_2_0(const std::vector<UInt_t>& permutation1, 
                               const std::vector<UInt_t>& permutation2);
    Int_t EqualPermutation_2_2(const std::vector<UInt_t>& permutation1, 
                               const std::vector<UInt_t>& permutation2);
    Int_t EqualPermutation_N_1(const std::vector<UInt_t>& permutation1, 
                               const std::vector<UInt_t>& permutation2);

    void  Print(const std::vector<UInt_t>& permutation);
    void  Print(const std::vector <std::vector <UInt_t> >& permutations);

private:
    
    Int_t  CalculatePermutations();

    void   initial_permutation(int size, int *permutation);
    Bool_t next_permutation(int size, int *permutation);
    void   initial_subset(int k, int *subset);
    Bool_t next_subset(int n, int k, int *subset);

    void  Skip_2_0(const std::vector <std::vector <UInt_t> >& permutation1, 
                   std::vector <std::vector <UInt_t> > &permutation2);
    void  Skip_2_2(const std::vector <std::vector <UInt_t> >& permutation1, 
                   std::vector <std::vector <UInt_t> > &permutation2);

    std::vector<UInt_t> Rotate(const std::vector <UInt_t>& permutation, UInt_t digits);

    const Int_t m_SetQuantity;
    const Int_t m_SubsetQuantity;

    std::vector<UInt_t> m_Subset;
    std::vector <std::vector <UInt_t> > m_Permutations;
    std::vector <std::vector <UInt_t> > m_Combinations;
};

#endif
