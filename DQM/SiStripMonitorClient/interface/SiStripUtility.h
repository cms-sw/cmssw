#ifndef SiStripUtility_H
#define SiStripUtility_H

/** \class SiStripUtility
 * *
 *  Class that handles the SiStrip Quality Tests
 * 
 *  $Date: 2006/05/21 19:42:07 $
 *  $Revision: 1.1 $
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
 static bool checkME(string element, string name, string& full_path);
 static int getMEList(string name, string& dir_path, vector<string>& me_names);

 static void split(const string& str, vector<string>& tokens, 
             const string& delimiters=" ");
};

#endif
