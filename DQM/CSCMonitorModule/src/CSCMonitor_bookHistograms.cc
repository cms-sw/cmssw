#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"


ME_List CSCMonitor::bookCommon(int nodeNumber) 
{
	std::string dir = "CSC/Common";
//	dbe->setCurrentFolder(dir);
//	string prefix = Form("EMU_%d", nodeNumber);
	string prefix = "EMU";
	ME_List commonMEs;
	ME_List_iterator itr;
	for (itr = commonMEfactory.begin(); itr != commonMEfactory.end(); ++itr) {
		// dbe->setCurrentFolder(dir);
		CSCMonitorObject * obj = new CSCMonitorObject(*itr->second);
		// obj->setPrefix(prefix+"_");
		commonMEs[obj->getName()] = obj;
		dbe->setCurrentFolder(dir+"/"+obj->getFolder());
		obj->Book(dbe);
		// commonMEs.insert(pair<string, CSCMonitorObject>(obj.getName(),obj));
	}

//	hw_tree.insert(hw_tree.begin(), prefix);	
	return commonMEs;
}

ME_List CSCMonitor::bookDDU(int dduNumber) 
{
	
	string prefix = Form("DDU_%02d", dduNumber);
	string dir = "CSC/"+prefix;
	dbe->setCurrentFolder(dir);
 	ME_List dduMEs;
        ME_List_iterator itr;
	
	for (itr = dduMEfactory.begin(); itr != dduMEfactory.end(); ++itr) {
		// dbe->setCurrentFolder(dir);
		CSCMonitorObject* obj = new CSCMonitorObject(*itr->second);
		// obj->setPrefix(prefix+"_");
		dduMEs[obj->getName()] = obj;
		dbe->setCurrentFolder(dir+"/"+obj->getFolder());
		obj->Book(dbe);
		// dduMEs.insert(pair<string, CSCMonitorObject>(obj.getName(),obj));
	}
/*	
	string node = Form("EMU_%d", nodeNumber);
	tree<std::string>::iterator root = find(hw_tree.begin(), hw_tree.end(), node);
	if (root != hw_tree.end()) {
		hw_tree.append_child(root, prefix);
	}
*/
	return dduMEs;
}

ME_List CSCMonitor::bookChamber(int chamberID) 
{
	int crate = (chamberID >> 4) & 0xFF;
	int slot = chamberID & 0xF;
	string prefix = Form("CSC_%03d_%02d", crate, slot);
	string dir = "CSC/"+prefix;
	dbe->setCurrentFolder(dir);
	ME_List chamberMEs;
        ME_List_iterator itr;

	for (itr = chamberMEfactory.begin(); itr != chamberMEfactory.end(); ++itr) {
		// dbe->setCurrentFolder(dir);
		CSCMonitorObject* obj = new CSCMonitorObject(*itr->second);
		// obj->setPrefix(prefix+"_");
		chamberMEs[obj->getName()] = obj;
		dbe->setCurrentFolder(dir+"/"+obj->getFolder());
		obj->Book(dbe);
		// chamberMEs.insert(pair<string, CSCMonitorObject>(obj.getName(),obj));
	}
/*
	string ddu_str = Form("DDU_%02d", dduNumber);
	tree<std::string>::iterator ddu = find(hw_tree.begin(), hw_tree.end(), ddu_str);
        if (ddu != hw_tree.end()) {
                hw_tree.append_child(ddu, prefix);
        }
*/
	
	return chamberMEs;
}
