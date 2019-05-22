#ifndef TSinglePedEntry_h
#define TSinglePedEntry_h

/**
 * \file TSinglePedEntry.h
 * \brief Transient container for a single entry in pedestal offset studies
 *
 * $Date:
 * $Revision:
 * \author P. Govoni (pietro.govoni@cernNOSPAM.ch)
 */

class TSinglePedEntry {
public:
  //! ctor
  TSinglePedEntry();
  //! copy ctor
  TSinglePedEntry(const TSinglePedEntry &orig);
  //! dtor
  ~TSinglePedEntry();

  //! add a single value
  void insert(const int &pedestal);
  //! get the average of the inserted values
  double average() const;
  //! get the RMS of the inserted values
  double RMS() const;
  //! get the squared RMS of the inserted values
  double RMSSq() const;

private:
  //! squared sum of entries
  int m_pedestalSqSum;
  //! sum of entries
  int m_pedestalSum;
  //! number of entries
  int m_entries;
};

#endif
