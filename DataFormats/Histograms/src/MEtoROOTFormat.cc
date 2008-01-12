/** \file MEtoROOTFormat.cc
 *  
 *  See header file for description of class
 *  
 *
 *  $Date: 2007/12/05 05:37:14 $
 *  $Revision: 1.3 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "DataFormats/Histograms/interface/MEtoROOTFormat.h"
#include <iostream>
#include <stdlib.h>

template <>
void MEtoROOT<TH1F>::putMERootObject(std::vector<std::string> name,
				     std::vector<TagList> tags,
				     std::vector<TH1F> object)
{

  MERootObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MERootObject[i].name = name[i];
    MERootObject[i].tags = tags[i];
    MERootObject[i].object = object[i];
  }

  return;
}

template <>
void MEtoROOT<TH2F>::putMERootObject(std::vector<std::string> name,
				     std::vector<TagList> tags,
				     std::vector<TH2F> object)
{

  MERootObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MERootObject[i].name = name[i];
    MERootObject[i].tags = tags[i];
    MERootObject[i].object = object[i];
  }

  return;
}

template <>
void MEtoROOT<TH3F>::putMERootObject(std::vector<std::string> name,
				     std::vector<TagList> tags,
				     std::vector<TH3F> object)
{

  MERootObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MERootObject[i].name = name[i];
    MERootObject[i].tags = tags[i];
    MERootObject[i].object = object[i];
  }

  return;
}

template <>
void MEtoROOT<TProfile>::putMERootObject(std::vector<std::string> name,
					 std::vector<TagList> tags,
					 std::vector<TProfile> object)
{

  MERootObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MERootObject[i].name = name[i];
    MERootObject[i].tags = tags[i];
    MERootObject[i].object = object[i];
  }

  return;
}

template <>
void MEtoROOT<TProfile2D>::putMERootObject(std::vector<std::string> name,
					   std::vector<TagList> tags,
					   std::vector<TProfile2D> object)
{

  MERootObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MERootObject[i].name = name[i];
    MERootObject[i].tags = tags[i];
    MERootObject[i].object = object[i];
  }

  return;
}

template <>
void MEtoROOT<float>::putMERootObject(std::vector<std::string> name,
				      std::vector<TagList> tags,
				      std::vector<float> object)
{

  MERootObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MERootObject[i].name = name[i];
    MERootObject[i].tags = tags[i];
    MERootObject[i].object = object[i];
  }

  return;
}

template <>
void MEtoROOT<int>::putMERootObject(std::vector<std::string> name,
				    std::vector<TagList> tags,
				    std::vector<int> object)
{

  MERootObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MERootObject[i].name = name[i];
    MERootObject[i].tags = tags[i];
    MERootObject[i].object = object[i];
  }

  return;
}

template <>
void MEtoROOT<std::string>::putMERootObject(std::vector<std::string> name,
					    std::vector<TagList> tags,
					    std::vector<std::string> object)
{

  MERootObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MERootObject[i].name = name[i];
    MERootObject[i].tags = tags[i];
    MERootObject[i].object = object[i];
  }

  return;
}
