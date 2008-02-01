/** \file MEtoEDMFormat.cc
 *  
 *  See header file for description of class
 *  
 *
 *  $Date: 2008/01/25 23:14:08 $
 *  $Revision: 1.5 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"
#include <iostream>
#include <stdlib.h>

template <>
void MEtoEDM<TH1F>::putMEtoEdmObject(std::vector<std::string> name,
				     std::vector<TagList> tags,
				     std::vector<TH1F> object)
{

  MEtoEdmObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MEtoEdmObject[i].name = name[i];
    MEtoEdmObject[i].tags = tags[i];
    MEtoEdmObject[i].object = object[i];
  }

  return;
}

template <>
void MEtoEDM<TH2F>::putMEtoEdmObject(std::vector<std::string> name,
				     std::vector<TagList> tags,
				     std::vector<TH2F> object)
{

  MEtoEdmObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MEtoEdmObject[i].name = name[i];
    MEtoEdmObject[i].tags = tags[i];
    MEtoEdmObject[i].object = object[i];
  }

  return;
}

template <>
void MEtoEDM<TH3F>::putMEtoEdmObject(std::vector<std::string> name,
				     std::vector<TagList> tags,
				     std::vector<TH3F> object)
{

  MEtoEdmObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MEtoEdmObject[i].name = name[i];
    MEtoEdmObject[i].tags = tags[i];
    MEtoEdmObject[i].object = object[i];
  }

  return;
}

template <>
void MEtoEDM<TProfile>::putMEtoEdmObject(std::vector<std::string> name,
					 std::vector<TagList> tags,
					 std::vector<TProfile> object)
{

  MEtoEdmObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MEtoEdmObject[i].name = name[i];
    MEtoEdmObject[i].tags = tags[i];
    MEtoEdmObject[i].object = object[i];
  }

  return;
}

template <>
void MEtoEDM<TProfile2D>::putMEtoEdmObject(std::vector<std::string> name,
					   std::vector<TagList> tags,
					   std::vector<TProfile2D> object)
{

  MEtoEdmObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MEtoEdmObject[i].name = name[i];
    MEtoEdmObject[i].tags = tags[i];
    MEtoEdmObject[i].object = object[i];
  }

  return;
}

template <>
void MEtoEDM<float>::putMEtoEdmObject(std::vector<std::string> name,
				      std::vector<TagList> tags,
				      std::vector<float> object)
{

  MEtoEdmObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MEtoEdmObject[i].name = name[i];
    MEtoEdmObject[i].tags = tags[i];
    MEtoEdmObject[i].object = object[i];
  }

  return;
}

template <>
void MEtoEDM<int>::putMEtoEdmObject(std::vector<std::string> name,
				    std::vector<TagList> tags,
				    std::vector<int> object)
{

  MEtoEdmObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MEtoEdmObject[i].name = name[i];
    MEtoEdmObject[i].tags = tags[i];
    MEtoEdmObject[i].object = object[i];
  }

  return;
}

template <>
void MEtoEDM<TString>::putMEtoEdmObject(std::vector<std::string> name,
					std::vector<TagList> tags,
					std::vector<TString> object)
{

  MEtoEdmObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MEtoEdmObject[i].name = name[i];
    MEtoEdmObject[i].tags = tags[i];
    MEtoEdmObject[i].object = object[i];
  }

  return;
}
