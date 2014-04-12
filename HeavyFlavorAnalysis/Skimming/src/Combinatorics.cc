/*
 * Combinatorics.cpp
 *
 * 03/04/2006 kasselmann@physik.rwth-aachen.de
 * 19/08/2007 giffels@physik.rwth-aachen.de
 *
 */
//framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Own header
#include "HeavyFlavorAnalysis/Skimming/interface/Combinatorics.h"

using namespace std;

// **************************************************************************
//  Class Constructor
// **************************************************************************
Combinatorics::Combinatorics(Int_t SetQuantity, Int_t SubsetQuantity) :

  m_SetQuantity(SetQuantity),
  m_SubsetQuantity(SubsetQuantity)
{
  // Get permutations
  CalculatePermutations();
}


// **************************************************************************
//  Class Destructor
// **************************************************************************
Combinatorics::~Combinatorics() 
{}


// **************************************************************************
//  Get subset permutations 
// **************************************************************************
vector <vector <UInt_t> > Combinatorics::GetPermutations() 
{
  return m_Permutations;
}

// **************************************************************************
//  Calculate all subset permutations 
// **************************************************************************
Int_t Combinatorics::CalculatePermutations() 
{
  if (m_SetQuantity < 1 || m_SubsetQuantity < 1 || (m_SetQuantity < m_SubsetQuantity)) 
    {
      edm::LogWarning("Combinatorics") << "[Combinatorics] No valid choice of set or subset!" << endl;
      return -1;
    }
  
  Int_t* currentSubset = new Int_t[m_SubsetQuantity];
  Int_t* currentMapping = new Int_t[m_SetQuantity];

  initial_subset(m_SubsetQuantity, currentSubset);    
  do
    {
      initial_permutation(m_SetQuantity, currentMapping);        
      do
	{
	  for (UShort_t i = 0; i < m_SubsetQuantity; i++) 
	    {
              m_Subset.push_back(currentSubset[currentMapping[i]]);
	    }
          m_Permutations.push_back(m_Subset);
          m_Subset.clear();
	}
      while (next_permutation(m_SubsetQuantity, currentMapping));
    }
  while (next_subset(m_SetQuantity, m_SubsetQuantity, currentSubset));

  delete[] currentSubset;
  delete[] currentMapping;

  return 0;
}


// **************************************************************************
// Build initial permutation
// **************************************************************************
void Combinatorics::initial_permutation(int size, int *permutation)
{
  for (int i = 0; i < size; i++) 
    {
      permutation[i] = i;
    }
}


// **************************************************************************
// Build initial subset
// **************************************************************************
void Combinatorics::initial_subset(int k, int *subset)
{
  for (int i = 0; i < k; i++) 
    {
      subset[i] = i;
    }
}


// **************************************************************************
// Get next permutation if a next permutation exists
// **************************************************************************
Bool_t Combinatorics::next_permutation(int size, int *permutation)
{
  int i, j, k;
  if (size < 2) return false;
  i = size - 2;
  
  while ((permutation[i] > permutation[i+1]) && (i != 0)) 
    {
      i--;
    }
  if ((i == 0) && (permutation[0] > permutation[1])) return false;
  
  k = i + 1;
  for (j = i + 2; j < size; j++ ) 
    {
      if ((permutation[j] > permutation[i]) && (permutation[j] < permutation[k])) 
	{
	  k = j;
        }
    }
  
  // swap i and k 
  {
    int tmp = permutation[i];
    permutation[i] = permutation[k];
    permutation[k] = tmp;
  }
  
  for (j = i + 1; j <= ((size + i) / 2); j++) 
    {
      int tmp = permutation[j];
      permutation[j] = permutation[size + i - j];
      permutation[size + i - j] = tmp;
    }

  // return whether a next permutation exists
  return true;
}


// **************************************************************************
// Get next subset if a next subset exists
// 
// n: the size of the set
// k: the size of the subset
// **************************************************************************
Bool_t Combinatorics::next_subset(int n, int k, int *subset)
{
  int i;
  int j;
  int jsave;
  bool done;
  
  if ( subset[0] < n-k )
    {
      done = false;
      jsave = k-1;
      for ( j = 0; j < k-1; j++ )
        {
	  if ( subset[j] + 1 < subset[j+1] )
            {
	      jsave = j;
	      break;
            }
        }
      for ( i = 0; i < jsave; i++ ) subset[i] = i;
      subset[jsave] = subset[jsave] + 1;
    }
  else
    {
      done = true;
    }
  // return whether a next subset exists
  return !done;
}


// **************************************************************************
//  Get all subset combinations
// **************************************************************************
vector < vector <UInt_t> > Combinatorics::GetCombinations()
{
  if (m_Permutations.size() == 0) 
    {
      LogDebug("Combinatorics") << "Nothing to do." << endl;
      return m_Combinations;
    }
  
  m_Combinations.push_back(m_Permutations.at(0));

  for (UInt_t i = 1; i < m_Permutations.size(); i++)
    {
      if (!EqualPermutation(m_Combinations.back(), m_Permutations.at(i))) 
        {
          m_Combinations.push_back(m_Permutations.at(i));
        }
    }
  return m_Combinations;
}


// **************************************************************************
// Returns true if two permutations of four "int" are equal
// (Equal means e.g.: 0123 = 1023 = 0132 = 1032)
// **************************************************************************
Int_t Combinatorics::EqualPermutation(const vector<UInt_t>& p1, const vector <UInt_t>& p2)
{
  if (p1.size() != p2.size())
    {
      edm::LogWarning("Combinatorics") << "[Combinatorics::EqualPermutation] permutations have different size!" << endl;
      return -1;
    }
  
  Float_t p1_sum = 0.0;
  Float_t p2_sum = 0.0;

  // Check whether permutations are equal (2^index)
  for (UInt_t i = 0; i < p1.size(); i++) p1_sum += (1 << p1.at(i));
  for (UInt_t i = 0; i < p2.size(); i++) p2_sum += (1 << p2.at(i));
  
  return (p1_sum == p2_sum ? 1 : 0);
}


// **************************************************************************
//  Get combinations: 4 out of n 
// 
//  (The order of the first and second two is not important!
//   0123 = 1023 = 0132 = 1032 are equal therefore)
// **************************************************************************
vector < vector <UInt_t> > Combinatorics::GetCombinations_2_2()
{
  // combination vector returned
  vector< vector <UInt_t> > FinalCombinations; 

  if (m_Permutations.size() == 0) 
    {
      LogDebug("Combinatorics") << "[Combinatorics::GetCombinations_2_2] Nothing to do." << endl;
      return FinalCombinations;
    }
  
  // So far only for subsets of four indices
  if (m_SubsetQuantity != 4) 
    {
      edm::LogWarning("Combinatorics") << "[Combinatorics::GetCombinations_2_2] Subset must be 4." << endl;
      return FinalCombinations;
    }

  // Skip specific permutations
  Skip_2_2(m_Permutations, FinalCombinations);

  return FinalCombinations;
}


// **************************************************************************
//  Get combinations: 4 out of n 
// 
//  (The order of the last two is important only: 
//   0123 = 1023 are equal therefore)
// **************************************************************************
vector < vector <UInt_t> > Combinatorics::GetCombinations_2_0()
{
  // combination vector returned
  vector< vector <UInt_t> > FinalCombinations; 

  if (m_Permutations.size() == 0) 
    {
      LogDebug("Combinatorics") << "[Combinatorics::GetCombinations_2_0] Nothing to do." << endl;
      return FinalCombinations;
    }
  
  // So far only for subsets of four indices
  if (m_SubsetQuantity != 4) 
    {
      edm::LogWarning("Combinatorics") << "[Combinatorics::GetCombinations_2_0] Subset must be 4." << endl;
      return FinalCombinations;
    }

  // Skip specific permutations
  Skip_2_0(m_Permutations, FinalCombinations);

  return FinalCombinations;
}


// **************************************************************************
// Skip permutation from p1 if already existing in p2
// **************************************************************************
void Combinatorics::Skip_2_0(const vector <vector <UInt_t> >& p1, 
                             vector <vector <UInt_t> > &p2)
{
  Bool_t Skip = kFALSE;

  p2.push_back(p1.at(0));
  
  for (UShort_t i = 1; i < p1.size(); i++)
    {
      for (UShort_t j = 0; j < p2.size(); j++)
        {
          if (EqualPermutation_2_0(p1.at(i), p2.at(j))) 
            {
              Skip = kTRUE;
            }
        }
      if (!Skip) p2.push_back(p1.at(i));

      Skip = kFALSE;
    }
}


// **************************************************************************
// Skip permutation from p1 if already existing in p2
// **************************************************************************
void Combinatorics::Skip_2_2(const vector <vector <UInt_t> >& p1, 
                             vector <vector <UInt_t> > &p2)
{
  Bool_t Skip = kFALSE;

  p2.push_back(p1.at(0));
  
  for (UShort_t i = 1; i < p1.size(); i++)
    {
      for (UShort_t j = 0; j < p2.size(); j++)
        {
          if (EqualPermutation_2_2(p1.at(i), p2.at(j))) 
            {
              Skip = kTRUE;
            }
        }
      if (!Skip) p2.push_back(p1.at(i));

      Skip = kFALSE;
    }
}


// **************************************************************************
// Returns true if the two first digm_ of two permutations are equal
// e.g.: 0123 = 1023 
// **************************************************************************
Int_t Combinatorics::EqualPermutation_2_0(const vector<UInt_t>& p1, const vector<UInt_t>& p2)
{
  if (p1.size() < 2) 
    {
      edm::LogWarning("Combinatorics") << "[Combinatorics::EqualPermutation_2_0] permutation has wrong size!" << endl;
      return -1;
    }
  
  // Check whether permutations are equal
  if ( ((1 << p1.at(0)) + (1 << p1.at(1)) == (1 << p2.at(0)) + (1 << p2.at(1)) ) &&
       p1.at(2) == p2.at(2) &&  p1.at(3) == p2.at(3) )
    {
      return 1;
    }
  return 0;
}


// **************************************************************************
// Returns true if two permutations of four "int" are equal
// e.g.: 0123 = 1023 = 0132 = 1032
// **************************************************************************
Int_t Combinatorics::EqualPermutation_2_2(const vector<UInt_t>& p1, const vector<UInt_t>& p2)
{
  // Returns true if two permutations of four "int" are equal 
  // (equal means e.g.: 0123 = 1023 = 0132 = 1032)

  if (p1.size() != 4 && p2.size() != 4) 
    {
      edm::LogWarning("Combinatorics") << "[Combinatorics::EqualPermutationTwoByTwo] permutation(s) have wrong size!" << endl;
      return -1;
    }
  
  // Check whether permutations are equal (2^index)
  if ( ((1 << p1.at(0)) + (1 << p1.at(1)) == (1 << p2.at(0)) + (1 << p2.at(1)) ) && 
       ((1 << p1.at(2)) + (1 << p1.at(3)) == (1 << p2.at(2)) + (1 << p2.at(3)) ) ) 
    {
      return 1;
    }
  return 0;
}


// **************************************************************************
// Returns true if two permutations of four are "equal"
// e.g.: 0123 = 1023 
// **************************************************************************
Int_t Combinatorics::EqualPermutation_N_1(const vector<UInt_t>& p1,const vector<UInt_t>& p2)
{
  // Returns true if two permutations of four "int" are equal 
  // (equal means e.g.: 012 = 102)

  if (p1.size() !=  p2.size()) 
    {
      edm::LogWarning("Combinatorics") << "[Combinatorics::EqualPermutationTwoByTwo] permutation(s) have wrong size!" << endl;
      return -1;
    }
  
  return (EqualPermutation(p1, p2) && p1.back() == p2.back() ? 1 : 0);
}


// **************************************************************************
//  Get combinations "N by 1"
// **************************************************************************
vector < vector <UInt_t> > Combinatorics::GetCombinations_N_1()
{
  // Get combinations 
  m_Combinations.clear();
  GetCombinations();
  
  // combination vector returned
  vector < vector <UInt_t> > FinalCombinations;

  if (m_Combinations.size() == 0) 
    {
      LogDebug("Combinatorics") << "[Combinatorics::GetCombinationsThreeByOne] Nothing to do." << endl;
      return FinalCombinations;
    }
  
  for (UInt_t i = 0; i < m_Combinations.size(); i++)
    {
      vector <UInt_t> RotatingPermutation = m_Combinations.at(i);
      FinalCombinations.push_back(m_Combinations.at(i));

      for (UInt_t j = 1; j < RotatingPermutation.size(); j++)
        {
          FinalCombinations.push_back(Rotate(RotatingPermutation,j));
        }
    }
  return FinalCombinations;
}


// **************************************************************************
// Rotate permutation to the "left" by n digm_
// **************************************************************************
vector<UInt_t> Combinatorics::Rotate(const vector <UInt_t>& permutation, UInt_t digm_)
{
  vector<UInt_t> p;
  vector<UInt_t> tmp;
  
  if (permutation.size() <= digm_) 
    {
      edm::LogWarning("Combinatorics") << "[Combinatorics::Rotate] WARNING: More rotations than digm_ in permutation!" << endl;   
    }

  // Save the first i digm_
  for (UInt_t i = 0; i < digm_; i++) 
    {
      tmp.push_back(permutation.at(i));
    }
  for (UInt_t j = 0; j < permutation.size() - digm_; j++)
    {
      p.push_back(permutation.at(j + digm_));
    }
  for (UInt_t k = 0; k < digm_; k++) p.push_back(tmp.at(k));

  return p;
}

  
// **************************************************************************
//  Print one permutation
// **************************************************************************
void Combinatorics::Print(const vector<UInt_t>& p)
{
  // Print permutations
  for (UShort_t i = 0; i < p.size(); i++)
    {
      LogDebug("Combinatorics") << (p.at(i));
    }
  LogDebug("Combinatorics") << endl;
}


// **************************************************************************
//  Print permutations 
// **************************************************************************
void Combinatorics::Print(const vector <vector <UInt_t> >& p)
{
  LogDebug("Combinatorics") << "**************" << endl;
  LogDebug("Combinatorics") << "Permutations: " << p.size() << endl;

  // Print permutations
  for (UShort_t i = 0; i < p.size(); i++)
    {
      for(UShort_t j = 0; j < (p.at(0)).size(); j++) LogDebug("Combinatorics") << (p.at(i)).at(j);
      LogDebug("Combinatorics") << endl;
    }
}











