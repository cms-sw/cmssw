/** \file MEtoEDMFormat.cc
 *  
 *  See header file for description of class
 *  
 *
 *  $Date: 2008/02/01 01:19:22 $
 *  $Revision: 1.1 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"
#include <iostream>
#include <stdlib.h>

template <>
void MEtoEDM<TH1F>::putMEtoEdmObject(std::vector<std::string> const &name,
				     std::vector<TagList> const &tags,
				     std::vector<TH1F> const &object)
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
void MEtoEDM<TH2F>::putMEtoEdmObject(std::vector<std::string> const &name,
				     std::vector<TagList> const &tags,
				     std::vector<TH2F> const &object)
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
void MEtoEDM<TH3F>::putMEtoEdmObject(std::vector<std::string> const &name,
				     std::vector<TagList> const &tags,
				     std::vector<TH3F> const &object)
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
void MEtoEDM<TProfile>::putMEtoEdmObject(std::vector<std::string> const &name,
					 std::vector<TagList> const &tags,
					 std::vector<TProfile> const &object)
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
void 
MEtoEDM<TProfile2D>::putMEtoEdmObject(std::vector<std::string> const &name,
				      std::vector<TagList> const &tags,
				      std::vector<TProfile2D> const &object)
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
void MEtoEDM<float>::putMEtoEdmObject(std::vector<std::string> const &name,
				      std::vector<TagList> const &tags,
				      std::vector<float> const &object)
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
void MEtoEDM<int>::putMEtoEdmObject(std::vector<std::string> const &name,
				    std::vector<TagList> const &tags,
				    std::vector<int> const &object)
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
void MEtoEDM<TString>::putMEtoEdmObject(std::vector<std::string> const &name,
					std::vector<TagList> const &tags,
					std::vector<TString> const &object)
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
bool MEtoEDM<TH1F>::mergeProduct(MEtoEDM<TH1F> const &newMEtoEDM)
{

  const MEtoEdmObjectVector &newMEtoEDMObject = newMEtoEDM.getMEtoEdmObject();
  for (unsigned int i = 0; i < MEtoEdmObject.size(); ++i) {
    MEtoEdmObject[i].object.Add(&newMEtoEDMObject[i].object);
  }

  return true;

}

template <>
bool MEtoEDM<TH2F>::mergeProduct(MEtoEDM<TH2F> const &newMEtoEDM)
{

  const MEtoEdmObjectVector &newMEtoEDMObject = newMEtoEDM.getMEtoEdmObject();
  for (unsigned int i = 0; i < MEtoEdmObject.size(); ++i) {
    MEtoEdmObject[i].object.Add(&newMEtoEDMObject[i].object);
  }

  return true;

}

template <>
bool MEtoEDM<TH3F>::mergeProduct(MEtoEDM<TH3F> const &newMEtoEDM)
{

  const MEtoEdmObjectVector newMEtoEDMObject = newMEtoEDM.getMEtoEdmObject();
  for (unsigned int i = 0; i < MEtoEdmObject.size(); ++i) {
    MEtoEdmObject[i].object.Add(&newMEtoEDMObject[i].object);
  }

  return true;

}

template <>
bool MEtoEDM<TProfile>::mergeProduct(MEtoEDM<TProfile> const &newMEtoEDM)
{

  const MEtoEdmObjectVector &newMEtoEDMObject = newMEtoEDM.getMEtoEdmObject();
  for (unsigned int i = 0; i < MEtoEdmObject.size(); ++i) {
    MEtoEdmObject[i].object.Add(&newMEtoEDMObject[i].object);
  }

  return true;

}

template <>
bool MEtoEDM<TProfile2D>::mergeProduct(MEtoEDM<TProfile2D> const &newMEtoEDM)
{

  const MEtoEdmObjectVector &newMEtoEDMObject = newMEtoEDM.getMEtoEdmObject();
  for (unsigned int i = 0; i < MEtoEdmObject.size(); ++i) {
    MEtoEdmObject[i].object.Add(&newMEtoEDMObject[i].object);
  }

  return true;

}

template <>
bool MEtoEDM<float>::mergeProduct(MEtoEDM<float> const &newMEtoEDM)
{

  return true;

}

template <>
bool MEtoEDM<int>::mergeProduct(MEtoEDM<int> const &newMEtoEDM)
{

  return true;

}

template <>
bool MEtoEDM<TString>::mergeProduct(MEtoEDM<TString> const &newMEtoEDM)
{

  return true;

}
