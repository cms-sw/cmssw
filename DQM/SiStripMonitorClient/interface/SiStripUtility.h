#ifndef SiStripUtility_H
#define SiStripUtility_H

/** \class SiStripUtility
 * *
 *  Class that handles the SiStrip Quality Tests
 * 
 *  $Date: 2006/16/05 17:42:28 $
 *  $Revision: 0.0 $
 *  \author Suchandra Dutta
  */

#include <vector>
#include <fstream>
#include <string>
#include <map>

using namespace std;

class SiStripUtility
{
 public:
 
 static int getMEList(string name, vector<string>& values);
 bool getME(string element, string name, string& full_path);

 static void split(const string& str, vector<string>& tokens, 
             const string& delimiters=" ");
};

#endif
