#include <TMap.h>
#include <TObjString.h>
#include "IORawData/HcalTBInputService/src/CDFRunInfo.h"
#include <cstdlib>

const char* CDFRunInfo::RootVariableName = "CDFRunInfo";

CDFRunInfo::CDFRunInfo() {}

CDFRunInfo::CDFRunInfo(TFile* file) { load(file); }

const char* CDFRunInfo::get(const char* key) const {
  std::map<std::string, std::string>::const_iterator i = m_mapData.find(key);
  if (i == m_mapData.end())
    return nullptr;
  return i->second.c_str();
}

int CDFRunInfo::getInt(const char* key) const {
  const char* k = get(key);
  if (k == nullptr)
    return 0;
  return atoi(k);
}

double CDFRunInfo::getDouble(const char* key) const {
  const char* k = get(key);
  if (k == nullptr)
    return 0;
  return atof(k);
}

int CDFRunInfo::getKeys(const char** buffer, int nbufs) {
  int j = 0;
  for (std::map<std::string, std::string>::const_iterator i = m_mapData.begin(); i != m_mapData.end() && j < nbufs;
       i++, j++) {
    buffer[j] = i->first.c_str();
  }
  return j;
}

bool CDFRunInfo::hasKey(const char* key) const {
  std::map<std::string, std::string>::const_iterator i = m_mapData.find(key);
  return (i != m_mapData.end());
}

void CDFRunInfo::setInfo(const char* key, const char* value) { m_mapData[key] = value; }

bool CDFRunInfo::load(TFile* f) {
  m_mapData.clear();
  if (f == nullptr)
    return false;
  TMap* pMap = (TMap*)f->Get(RootVariableName);
  if (pMap == nullptr)
    return false;
  TIterator* i = pMap->MakeIterator();
  TObject* o;

  while ((o = i->Next()) != nullptr) {
    std::string a(o->GetName());
    std::string b(pMap->GetValue(o)->GetName());
    m_mapData.insert(std::pair<std::string, std::string>(a, b));
  }
  return true;
}

void CDFRunInfo::store(TFile* f) {
  f->cd();
  TMap* myMap = new TMap();
  for (std::map<std::string, std::string>::iterator i = m_mapData.begin(); i != m_mapData.end(); i++) {
    myMap->Add(new TObjString(i->first.c_str()), new TObjString(i->second.c_str()));
  }
  myMap->SetName(RootVariableName);
  myMap->Write(RootVariableName, TObject::kSingleKey);
}

void CDFRunInfo::print() const {
  for (std::map<std::string, std::string>::const_iterator i = m_mapData.begin(); i != m_mapData.end(); i++)
    printf(" '%s' => '%s' \n", i->first.c_str(), i->second.c_str());
}
