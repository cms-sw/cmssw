#ifndef _generic_h
#define _generic_h_

#include <map>
#include <string>

std::string get_from_multimap(std::multimap<string, string> &mymap, std::string key)
{
  std::multimap<std::string, std::string>::iterator it;
  it = mymap.find(key);
  if (it != mymap.end())
    {
      return (it->second);
    }
  return "";
}

#endif
