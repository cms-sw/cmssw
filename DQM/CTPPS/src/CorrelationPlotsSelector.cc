/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Rafał Leszko (rafal.leszko@gmail.com)
*
****************************************************************************/

#include "DQM/CTPPS/interface/CorrelationPlotsSelector.h"

#include <QString>
#include <QStringList>

CorrelationPlotsSelector::CorrelationPlotsSelector(std::string filter)
{
  // creating vPlanes array
  for (int i = 0; i < 10; i++)
  {
    if (i % 2)
    {
      vPlanes[0][i] = vPlanes[1][i] = vPlanes[4][i] = vPlanes[5][i] = false;
      vPlanes[2][i] = vPlanes[3][i] = true;
    } else {
      vPlanes[0][i] = vPlanes[1][i] = vPlanes[4][i] = vPlanes[5][i] = true;
      vPlanes[2][i] = vPlanes[3][i] = false;
    }
  }

  ReadFilterString(filter);
}



void CorrelationPlotsSelector::ReadFilterString(std::string correlationPlotsFilter)
{
  QString filter(correlationPlotsFilter.c_str());
  filter = filter.simplified();
  QStringList filterList = filter.split(';');
  QString defaultFilter = filterList.first();
  defaultFilter = defaultFilter.simplified();
  filterList.pop_front();
  int pos = defaultFilter.indexOf('=');
  if (defaultFilter.left(7) != "default" || pos == -1) {
    printf("!!! The correlation filter string \"%s\" is not correct. Check it and try again.\n", correlationPlotsFilter.c_str());
    printf("!!! The correlation plots are not loaded\n");
    exit(1);
  }
  defaultFilter = defaultFilter.mid(pos+1).simplified();
  QStringList defaultFilterList = defaultFilter.split(',');
  for (QStringList::iterator iter = defaultFilterList.begin(); iter != defaultFilterList.end(); iter++) {
    if (iter->simplified() == "")
      continue;
    bool ok;
    unsigned int n;
    n = iter->simplified().toUInt(&ok);
    if (!ok) {
      printf("!!! The correlation filter string \"%s\" is not correct. Check it and try again.\n", correlationPlotsFilter.c_str());
      printf("!!! The correlation plots are not loaded\n");
      exit(1);
    }
    defaultPlaneIds.insert(n);
  }
  for (QStringList::iterator iter = filterList.begin(); iter != filterList.end(); iter++) {
    QString temp = iter->simplified();
    if (temp == "")
      continue;
    int pos = temp.indexOf('=');
    if (pos == -1) {
      printf("!!! The correlation filter string \"%s\" is not correct. Check it and try again.\n", correlationPlotsFilter.c_str());
      printf("!!! The correlation plots are not loaded\n");
      exit(1);
    }
    QString RPString = temp.left(pos).simplified();
    QString PlanesString = temp.mid(pos+1).simplified();
    bool ok;
    unsigned int RPId = RPString.toUInt(&ok);
    if (!ok) {
      printf("!!! The correlation filter string \"%s\" is not correct. Check it and try again.\n", correlationPlotsFilter.c_str());
      printf("!!! The correlation plots are not loaded\n");
      exit(1);
    }
    QStringList RPFilter = PlanesString.split(',');
    bool ifAny = false;
    for (QStringList::iterator i = RPFilter.begin(); i != RPFilter.end(); i++) {
      if (i->simplified() == "")
        continue;
      unsigned int PlaneId = i->simplified().toUInt(&ok);
      if (!ok) {
        printf("!!! The correlation filter string \"%s\" is not correct. Check it and try again.\n", correlationPlotsFilter.c_str());
        printf("!!! The correlation plots are not loaded\n");
        exit(1);
      }
      specifiedRPPlaneIds[RPId].insert(PlaneId);
      ifAny = true;
    }
    if (!ifAny)
      emptyRPs.insert(RPId);
  }

    /*
  // test of parsing correction
  printf("\ndefault:\n");
  for (std::set<int>::iterator iter = defaultPlaneIds.begin(); iter != defaultPlaneIds.end(); iter++)
    printf("\t%d\n", *iter);
  printf("\nRoman Pots specified:\n");
  for (std::map<int, std::set<int> >::iterator iter = specifiedRPPlaneIds.begin(); iter != specifiedRPPlaneIds.end(); iter++) {
    printf("\tRP %d\n", iter->first);
    for (std::set<int>::iterator i = iter->second.begin(); i != iter->second.end(); i++)
      printf("\t\t%d\n", *i);
  }
  printf("\nempty RP:\n");
  for (std::set<int>::iterator iter = emptyRPs.begin(); iter != emptyRPs.end(); iter++)
    printf("\t%d\n", *iter);
  //
  exit(1);
  */
}


bool CorrelationPlotsSelector::IfCorrelate(unsigned int DetId)
{
  unsigned int RPId = DetId / 10;
  unsigned int PlaneId = DetId % 10;
  return IfCorrelate(RPId, PlaneId);
}



bool CorrelationPlotsSelector::IfCorrelate(unsigned int RPId, unsigned int PlaneId)
{
  if (emptyRPs.find(RPId) == emptyRPs.end() && (specifiedRPPlaneIds[RPId].find(PlaneId) != specifiedRPPlaneIds[RPId].end() 
      || (specifiedRPPlaneIds[RPId].empty() && defaultPlaneIds.find(PlaneId) != defaultPlaneIds.end())))
    return true;

  return false;
}



bool CorrelationPlotsSelector::IfTwoCorrelate(unsigned int DetId1, unsigned int DetId2)
{
  unsigned int RPId1 = (DetId1 / 10) % 10;
  unsigned int RPId2 = (DetId2 / 10) % 10;
  unsigned int PlaneId1 = DetId1 % 10;
  unsigned int PlaneId2 = DetId2 % 10;

  return IfTwoCorrelate(RPId1, PlaneId1, RPId2, PlaneId2);
}



bool CorrelationPlotsSelector::IfTwoCorrelate(unsigned int RPId1, unsigned int PlaneId1, unsigned int RPId2, unsigned int PlaneId2)
{
  if (RPId1 > 5 || RPId2 > 5 || PlaneId1 > 9 || PlaneId2 > 9)
  {
    printf("ERROR: while selection plots for corelation diagram, wrong DetId\n");
    return false;
  }

  // only vPlane with vPlane & uPlane with uPlane
  if (!((vPlanes[RPId1][PlaneId1] && vPlanes[RPId2][PlaneId2]) || (!vPlanes[RPId1][PlaneId1] && !vPlanes[RPId2][PlaneId2])))
    return false;
  

  // remove up & bottom Roman Pots combinations
  if (((RPId1 == 0 || RPId1 == 4) && (RPId2 == 1 || RPId2 == 5)) || ((RPId1 == 1 || RPId1 == 5) && (RPId2 == 0 || RPId2 == 4)))
    return false;

  return true;
}
