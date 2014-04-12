
#include "CalibCalorimetry/EcalPedestalOffsets/interface/TSinglePedEntry.h"
#include <iostream>
#include <cmath>

TSinglePedEntry::TSinglePedEntry () : 
  m_pedestalSqSum (0),
  m_pedestalSum (0),
  m_entries (0)
{
//  std::cout << "[TSinglePedEntry][ctor]" << std::endl ;
}


TSinglePedEntry::~TSinglePedEntry ()
{}


TSinglePedEntry::TSinglePedEntry (const TSinglePedEntry & orig) 
{
  m_pedestalSqSum = orig.m_pedestalSqSum ;
  m_pedestalSum = orig.m_pedestalSum ;
  m_entries = orig.m_entries ;
} 
  

void TSinglePedEntry::insert (const int & pedestal) 
{
  m_pedestalSqSum += pedestal * pedestal ;
  m_pedestalSum += pedestal ;
  ++m_entries ;
}


double TSinglePedEntry::average () const
{
  if (!m_entries) return -1;
  return static_cast<double>(m_pedestalSum) / m_entries ;
}


double TSinglePedEntry::RMS () const
{
  if (!m_entries) return -1;
  return sqrt (RMSSq ())  ;
}

double TSinglePedEntry::RMSSq () const
{
/*
  std::cout << "[TSinglePedEntry][minchia] " << m_pedestalSum
            << "\t" << m_pedestalSum
            << "\t" << m_entries << std::endl ;   // FIXME
*/            
  if (!m_entries) return -1;
  double num = 1./static_cast<double>(m_entries) ;
  double output = m_pedestalSqSum * num - m_pedestalSum * num * m_pedestalSum * num ;
  return output ;
}


