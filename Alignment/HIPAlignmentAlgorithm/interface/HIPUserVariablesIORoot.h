#ifndef HIPUserVariablesIORoot_H
#define HIPUserVariablesIORoot_H

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORootBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentUserVariablesIO.h"

/** concrete class for ROOT based IO of AlignmentUserVariables */

class HIPUserVariablesIORoot : public AlignmentIORootBase, public AlignmentUserVariablesIO {
public:
  using Alignables = align::Alignables;

  /** constructor */
  HIPUserVariablesIORoot();

  /** write user variables */
  void writeHIPUserVariables(const Alignables& alivec, const char* filename, int iter, bool validCheck, int& ierr);

  /** read user variables */
  std::vector<AlignmentUserVariables*> readHIPUserVariables(const Alignables& alivec,
                                                            const char* filename,
                                                            int iter,
                                                            int& ierr);

private:
  /** write AlignmentParameters of one Alignable */
  int writeOne(Alignable* ali) override;

  /** read AlignmentParameters of one Alignable */
  AlignmentUserVariables* readOne(Alignable* ali, int& ierr) override;

  /** open IO */
  int open(const char* filename, int iteration, bool writemode) override {
    newopen = true;
    return openRoot(filename, iteration, writemode);
  }

  /** close IO */
  int close(void) override { return closeRoot(); }

  // helper functions

  int findEntry(unsigned int detId, int comp);
  void createBranches(void) override;
  void setBranchAddresses(void) override;

  // data members

  static const int nparmax = 19;

  /** alignment parameter tree */
  int ObjId;
  unsigned int Id;
  int Nhit, Nparj, Npare;
  int DataType;
  double Jtvj[nparmax * (nparmax + 1) / 2];
  double Jtve[nparmax];
  double AlignableChi2;
  unsigned int AlignableNdof;
  double Par[nparmax];
  double ParError[nparmax];

  bool newopen;
  typedef std::map<std::pair<int, int>, int> treemaptype;
  treemaptype treemap;
};

#endif
