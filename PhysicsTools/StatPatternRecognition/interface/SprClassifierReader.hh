// File and Version Information:
//      $Id: SprClassifierReader.hh,v 1.4 2007/05/25 17:59:17 narsky Exp $
//
// Description:
//      Class SprClassifierReader :
//          Framework for reading stored classifiers from an input stream.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2006              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprClassifierReader_HH
#define _SprClassifierReader_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include <string>
#include <istream>
#include <vector>
#include <map>
#include <utility>

class SprAbsClassifier;
class SprAbsTrainedClassifier;
class SprStdBackprop;
class SprTrainedTopdownTree;
class SprTrainedDecisionTree;
class SprTrainedFisher;
class SprTrainedLogitR;
class SprTrainedBinarySplit;
class SprAdaBoost;
class SprBagger;
class SprTrainedCombiner;

class SprAbsFilter;
class SprAbsTwoClassCriterion;
class SprIntegerBootstrap;
class SprTrainedNode;


class SprClassifierReader
{
public:
  virtual ~SprClassifierReader() {}

  SprClassifierReader() {}

  // Reads trained classifier from a file or from an input stream.
  static SprAbsTrainedClassifier* readTrained(const char* filename,
					      int verbose=0);
  static SprAbsTrainedClassifier* readTrained(std::istream& input,
					      int verbose=0);

  // Reads trained classifier of specified type. Returns ownership.
  // Upon failure returns null pointer.
  template<class T> static T* readTrained(const char* filename, 
					  const char* classifier,
					  int verbose=0);
  template<class T> static T* readTrained(std::istream& input, 
					  const char* classifier,
					  int verbose=0);

  // Sets the trainable classifier configuration 
  // to the one from the configuration file. Return false upon failure.
  static bool readTrainable(const char* filename, 
			    SprAbsClassifier* classifier,
			    int verbose=0);
  static bool readTrainable(std::istream& input, 
			    SprAbsClassifier* classifier,
			    int verbose=0);

  // Read classifier configurations from an input stream.
  static bool readTrainableConfig(std::istream& input,
				  unsigned& lineCounter,
				  SprAbsFilter* data,
				  bool discreteTree, 
				  bool mixedNodesTree,
				  bool fastSortTree,
				  std::vector<SprAbsTwoClassCriterion*>& crits,
				  std::vector<SprIntegerBootstrap*>& bstraps,
				  std::vector<SprAbsClassifier*>& classifiers,
				  std::vector<SprCCPair>& ccPairs,
				  bool readOneEntry=false);

private:
  friend class SprMultiClassReader;

  /*
    Reads classifier name.
  */
  static std::string readClassifierName(std::istream& input);

  /*
    Reads variables for the trained classifier at the end of input.
  */
  static bool readVars(std::istream& input, 
		       std::vector<std::string>& vars,
		       unsigned& lineCounter);

  /*
    Reads trained classifier from random position in a stream.
  */
  static SprAbsTrainedClassifier* readTrainedFromStream(std::istream& input,
					       const std::string& classifier,
							unsigned& lineCounter);

  /*
    Reads trained classifier of known type from random position in a stream.
  */
  static SprAbsTrainedClassifier* readSelectedTrained(std::istream& input,
					       const std::string& classifier,
						      unsigned& lineCounter);

  // specific reader implementations
  static bool readStdBackprop(std::istream& input, 
			      SprStdBackprop* trainable, 
			      unsigned& lineCounter);
  static SprTrainedTopdownTree* readTopdownTree(std::istream& input, 
						unsigned& lineCounter);
  static SprTrainedDecisionTree* readDecisionTree(std::istream& input, 
						  unsigned& lineCounter);
  static SprTrainedFisher* readFisher(std::istream& input, 
				      unsigned& lineCounter);
  static SprTrainedLogitR* readLogitR(std::istream& input, 
				      unsigned& lineCounter);
  static SprTrainedBinarySplit* readBinarySplit(std::istream& input, 
						unsigned& lineCounter);
  static bool readAdaBoost(std::istream& input, 
			   SprAdaBoost* trainable, 
			   unsigned& lineCounter);
  static bool readBagger(std::istream& input, 
			 SprBagger* trainable, 
			 unsigned& lineCounter);
  static SprTrainedCombiner* readCombiner(std::istream& input,
					  unsigned& lineCounter);

  // various helper typedefs
  typedef std::map<int,std::pair<SprTrainedNode*,std::pair<int,int> > > 
  SprTopdownNodeMap;

  // various helper methods
  static bool addTopdownTreeNode(SprTrainedNode* node, 
				 int dau1, int dau2,
				 SprTopdownNodeMap& mapped);
  static bool makeTopdownTreeNodeList(const SprTopdownNodeMap& mapped,
				      std::vector<const SprTrainedNode*>& 
				      nodes);
  static void prepareTopdownTreeExit(const SprTopdownNodeMap& mapped);
};

#endif
