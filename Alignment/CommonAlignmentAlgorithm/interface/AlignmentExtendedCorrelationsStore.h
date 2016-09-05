#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentExtendedCorrelationsStore_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentExtendedCorrelationsStore_h

/// This class manages the storage and retrieval of correlations between Alignables
/// for the AlignmentParameterStore. This implementation does not stores the entries
/// of the "big covariance matrix" itself, but the statistical correlations, i.e.
/// R_ij=C_ij/sqrt(C_ii*C_jj) rather than C_ij.
///
/// If a correlation exceeds a certain value (especially corrupted correlations with
/// an absolute value bigger than 1) it is downweighted.


#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentCorrelationsStore.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentExtendedCorrelationsEntry.h"

namespace edm { class ParameterSet; }

class AlignmentExtendedCorrelationsStore : public AlignmentCorrelationsStore
{

public:

  typedef AlignmentExtendedCorrelationsEntry ExtendedCorrelationsEntry;
  typedef std::map< Alignable*, ExtendedCorrelationsEntry > ExtendedCorrelationsTable;
  typedef std::map< Alignable*, ExtendedCorrelationsTable* > ExtendedCorrelations;

  AlignmentExtendedCorrelationsStore( const edm::ParameterSet& config );

  virtual ~AlignmentExtendedCorrelationsStore( void ) {}

  /// Write correlations directly to the covariance matrix starting at the
  /// given position. Indices are assumed to start from 0.
  virtual void correlations( Alignable* ap1, Alignable* ap2,
			     AlgebraicSymMatrix& cov, int row, int col ) const override;

  /// Get correlations directly from the given position of the covariance
  /// matrix and store them. Indices are assumed to start from 0.
  virtual void setCorrelations( Alignable* ap1, Alignable* ap2,
				const AlgebraicSymMatrix& cov, int row, int col ) override;

  /// Set correlations without checking whether the maximum
  /// number of updates has already been reached.
  virtual void setCorrelations( Alignable* ap1, Alignable* ap2,	AlgebraicMatrix& mat ) override;

  /// Get correlations.
  virtual void getCorrelations( Alignable* ap1, Alignable* ap2,	AlgebraicMatrix& mat ) const;

  /// Check whether correlations are stored for a given pair of alignables.
  virtual bool correlationsAvailable( Alignable* ap1, Alignable* ap2 ) const override;

  /// Reset correlations.
  virtual void resetCorrelations( void ) override;

  /// Get number of stored correlations.
  virtual unsigned int size( void ) const override;

private:

  void fillCorrelationsTable( Alignable* ap1, Alignable* ap2,
                              ExtendedCorrelationsTable* table,
                              const AlgebraicSymMatrix& cov,
                              int row, int col, bool transpose );

  void fillCovariance( Alignable* ap1, Alignable* ap2, const ExtendedCorrelationsEntry& entry,
                       AlgebraicSymMatrix& cov, int row, int col ) const;

  void fillCovarianceT( Alignable* ap1, Alignable* ap2, const ExtendedCorrelationsEntry& entry,
                        AlgebraicSymMatrix& cov, int row, int col ) const;

  void readFromCovariance( Alignable* ap1, Alignable* ap2, ExtendedCorrelationsEntry& entry,
                           const AlgebraicSymMatrix& cov, int row, int col );

  void readFromCovarianceT( Alignable* ap1, Alignable* ap2, ExtendedCorrelationsEntry& entry,
                            const AlgebraicSymMatrix& cov, int row, int col );

  void resizeCorruptCorrelations( ExtendedCorrelationsEntry& entry, double maxCorr );

  ExtendedCorrelations theCorrelations;

  int theMaxUpdates;
  double theCut;
  double theWeight;

};

#endif
