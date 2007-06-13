#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
#include <fstream>

void CSCMonitor::saveImages(std::string path, std::string format, int width, int height)
{

  fBusy = true;
  LOG4CPLUS_INFO(logger_, "Saving MEs as "<<format<< " images to " << path)
    if (format == "") { format = DEFAULT_IMAGE_FORMAT; }

  std::ofstream tree_items;

  gStyle->SetPalette(1,0);
  //  gStyle->SetTitleW(0.95);
  //  gStyle->SetTitleH(0.085);
 
  TString command = Form("mkdir -p %s",path.c_str());
  gSystem->Exec(command.Data());

  string runNumber = "";

  tree_items.open((path+"/tree_items.js").c_str());
  tree_items << "var TREE_ITEMS = [\n"
	     << "    ['RunNumber" << runNumber << "', ''," << endl;


  std::map<std::string, ME_List >::iterator itr;
  ME_List_const_iterator h_itr;


  string imgfile = "";
  for (itr = MEs.begin(); itr != MEs.end(); ++itr) {
    string relpath = itr->first;
    string fullpath  = path+"/"+relpath;
    command = Form("mkdir -p %s",fullpath.c_str()); 
    gSystem->Exec(command.Data());
    tree_items << "            ['"<< itr->first <<"', '"<< itr->first <<"'," << endl;	
    for (h_itr = itr->second.begin(); h_itr != itr->second.end(); ++h_itr) {
      string fullname = h_itr->second->getFullName() + "." + format;
      string relname = relpath + "/" + fullname;
      imgfile = fullpath + "/"+ fullname;
      TCanvas *cnv = new TCanvas("cnv",fullname.c_str(), width, height);
      string cnv_param  ="";
      if ((cnv_param  = h_itr->second->getParameter("SetOptStat")) != "")
	gStyle->SetOptStat(cnv_param.c_str());
      cnv->cd();
      h_itr->second->Draw();
      LOG4CPLUS_INFO(logger_, imgfile);
      cnv->Print(imgfile.c_str());
      tree_items << "                    ['"<< h_itr->second->getTitle() << "','" << relname <<"']," << endl;
      delete cnv;
    }
    tree_items << "            ]," << endl;  
  }
  tree_items << "    ],\n"
	     << "];" << endl;
  tree_items.clear();
  tree_items.close();

  createTreePage(path);
  createTreeEngine(path);
  createTreeTemplate(path);
  createHTMLNavigation(path);

  LOG4CPLUS_INFO(logger_, "Done");
  fBusy = false;

}
/*
  void CSCMonitor::saveCanvasImages(std::string path, std::string format, int width, int height)
  {

  fBusy = true;
  LOG4CPLUS_WARN(logger_, "Saving MEs as "<<format<< " images to to " << path)
  if (format == "") { format = DEFAULT_IMAGE_FORMAT; }

  std::ofstream tree_items;
  std::ofstream csc_list;

  //  if (!gApplication)
  //    TApplication::CreateApplication();

  gStyle->SetPalette(1,0);

  TString command = Form("mkdir -p %s",path.c_str());
  gSystem->Exec(command.Data());

  string runNumber = "";
  tree_items.open((path+"/tree_items.js").c_str());
  tree_items << "var TREE_ITEMS = [\n"
  << "    ['RunNumber" << runNumber << "', ''," << endl;

  csc_list.open((path+"/csc_list.js").c_str());
  csc_list << "var CSC_LIST = [\n"
  << "    ['RunNumber" << runNumber << "'";

 
  std::map<std::string, ME_List>::iterator me_itr; 
  std::map<std::string, MECanvases_List>::iterator itr;
  MECanvases_List_const_iterator h_itr;

  vector<string> EMU_folders;
  vector<string> DDU_folders;
  vector<string> CSC_folders;
  for (itr = MECanvases.begin(); itr != MECanvases.end(); ++itr) {
  string folder = itr->first;
  if (folder.find("EMU") != string::npos) EMU_folders.push_back(folder);
  else if (folder.find("DDU") != string::npos) DDU_folders.push_back(folder);
  else if (folder.find("CSC") != string::npos) CSC_folders.push_back(folder);    
  }

  if (EMU_folders.size()) {
  csc_list << ",\n['EMU',[";
  for (uint32_t i=0; i<EMU_folders.size(); i++ ) {
  csc_list << "'"<< EMU_folders[i] << ((i==EMU_folders.size())?"'":"',");
  }
  csc_list << "]]";
  }
  if (DDU_folders.size()) {
  csc_list << ",\n['DDU',[";
  for (uint32_t i=0; i<DDU_folders.size(); i++ ) {
  csc_list << "'"<< DDU_folders[i] << ((i==DDU_folders.size())?"'":"',");
  }
  csc_list << "]]";
  }
 

  // Quick fix !!!TODO: Sorted output 
  if (CSC_folders.size()) {
  int crate=0;
  int slot=0;
  int cur_crate=-1;
  for (uint32_t i=0; i<CSC_folders.size(); i++ ) {
  string csc_ptrn = "CSC_%d_%d";
  if (sscanf(CSC_folders[i].c_str(),csc_ptrn.c_str(), &crate, &slot) == 2) {
  if (crate != cur_crate) {
  if (cur_crate>=0) csc_list << "]]";
  csc_list << ",\n['crate"<< crate << "', [";
  }
  csc_list << "'slot" << slot << "',";
  }
  cur_crate = crate;

  }
  csc_list << "]]";
  }

  string imgfile = "";
  for (itr = MECanvases.begin(); itr != MECanvases.end(); ++itr) {
  string relpath = itr->first;
  if (relpath.find("CSC_") != string::npos) {
  int crate=0;
  int slot=0;
  string csc_ptrn = "CSC_%d_%d";
  if (sscanf(relpath.c_str(),csc_ptrn.c_str(), &crate, &slot) == 2) {
  ostringstream st;
  st << "crate" << crate << "/slot" << slot;
  relpath =st.str();
  }
  }
  tree_items << "            ['"<< itr->first <<"', '"<< relpath <<"'," << endl;
  for (h_itr = itr->second.begin(); h_itr != itr->second.end(); ++h_itr) {
  string fullname = h_itr->second->getName() + "." + format;
  string relname = relpath + "/" + h_itr->second->getFolder() +"/" + fullname;
  string fullpath  = path+"/"+relpath + "/" + h_itr->second->getFolder();
  imgfile = fullpath + "/"+ fullname;
  command = Form("mkdir -p %s",fullpath.c_str());
  gSystem->Exec(command.Data());
  me_itr = MEs.find(itr->first);
  if (me_itr != MEs.end()) {
  h_itr->second->Draw(me_itr->second, width, height);
  LOG4CPLUS_WARN(logger_, imgfile);
  h_itr->second->Print(imgfile.c_str());
  }
  tree_items << "                    ['"<< h_itr->second->getTitle() << "','" << relname <<"']," << endl;
  }
  tree_items << "            ]," << endl;
  }
  tree_items << "    ],\n"
  << "];" << endl;
  tree_items.clear();
  tree_items.close();
  
  csc_list << "]\n"
  << "];" << endl;
  csc_list.clear();
  csc_list.close();

  createTreePage(path);
  createTreeEngine(path);
  createTreeTemplate(path);
  createHTMLNavigation(path);

  LOG4CPLUS_INFO(logger_, "Done");
  fBusy = false;
  }

*/

/*
  void CSCMonitor::SaveImagesFromROOTFile(std::string rootfile,std::string path, std::string format, int width, int height)
  {
  if (rootfile == "") { return;}
  LOG4CPLUS_INFO(logger_, "Generating Histo Images from " << rootfile << " and saving them to " << path);
  if (format == "") { format = DEFAULT_IMAGE_FORMAT; }
  TString imgfile = "";
  TString filename(rootfile.c_str());
  TFile f(rootfile.c_str());
  gStyle->SetPalette(1,0);
  gStyle->SetTitleW(0.95);
  gStyle->SetTitleH(0.085);



  filename.ReplaceAll(".root", "");
  filename.Replace(0, filename.Last('/')+1,"");
  //  TString command = Form("rmdir -rf %s/%s",path.c_str(), filename.Data());
  //  gSystem->Exec(command.Data());

  TIterator *keys = f.GetListOfKeys()->MakeIterator();

  TKey *key=0;
  while( ( key=(TKey*)keys->Next() ) ){
  if( strncmp(key->GetClassName(),"TCanvas",7) ) continue;
  TObject *obj = key->ReadObj();

  TString fullPath(obj->GetName()), basename, dirname;
  Ssiz_t pos=0;
  if( (pos = fullPath.Last('/')) >=0 ){
  basename = fullPath(pos+1,fullPath.Length()-pos-1);
  dirname  = fullPath(0,pos);			

  } else basename = fullPath;
  pos=0;
  while(
  basename[pos]=='1' ||
  basename[pos]=='2' ||
  basename[pos]=='3' ||
  basename[pos]=='4' ||
  basename[pos]=='5' ||
  basename[pos]=='6' ||
  basename[pos]=='7' ||
  basename[pos]=='8' ||
  basename[pos]=='9' ||
  basename[pos]=='0' ||
  basename[pos]=='_' ) pos++;

  if( pos > 0 ){
  TString location = basename(0,pos).Data();
  Ssiz_t space = location.First('_');
  TString sCrate(location(0,space));
  int crate = atoi(sCrate.Data());
  TString sSlot(location(space+1,pos-space-1));
  int slot  = atoi(sSlot.Data());
  fullPath = Form("%s/%s/crate%d/slot%d/%s",path.c_str(), filename.Data(),crate,slot,dirname.Data());
  } else {
  fullPath = Form("%s/%s/%s", path.c_str(), filename.Data(),dirname.Data());
  }
  TString command = Form("mkdir -p %s",fullPath.Data());
  gSystem->Exec(command.Data());
  // obj->Draw();
  ((TCanvas*)obj)->SetCanvasSize(width, height);
  imgfile = Form("%s/%s.%s", fullPath.Data(), basename(pos,basename.Length()-pos).Data(), format.c_str());
  LOG4CPLUS_INFO(logger_, imgfile.Data());	
  ((TCanvas*)obj)->Print(imgfile.Data());
  delete obj;
  }
  LOG4CPLUS_INFO(logger_, "Done...");
  }
*/
void CSCMonitor::createHTMLNavigation(std:: string path) 
{

  std::ofstream indexfile;
  indexfile.open((path+"/browse.html").c_str());
  
  indexfile << "<html>\n<head>\n"
	    << "<title>Emu DQM Web Client Interface</title>\n"
	    << "</head>\n"
	    << "<frameset cols=\"200,*\">\n"
	    << "	<frame src=\"tree.html\" name=\"tree\" id=\"tree\">\n"
	    << "	<frame name=\"frameset\" id=\"frameset\">\n"
	    << "</frameset>\n"
	    << "<body>\n"
	    << "</body>\n"
	    << "</html>" << endl;


  /*	
    indexfile << "<html>\n"
    //	    << "<meta http-equiv=\"refresh\" content=\"10; URL=index.html\">\n"
    << "<frameset rows=\"25%, 75%\">\n"
    << "<frameset cols=\"25%, 70%\">\n"
    << "<frame src=\"folders.html\">\n"
    << "<frame src=\"canvases.html\" name=\"canvases_list\">\n"
    << "</frameset>\n"
    << "<frame src=\"imageout.html\" name=\"imageout\">\n"
    << "</frameset>\n"
    << "</html>\n";
  */
  indexfile.clear();
  indexfile.close();
  /* 
     indexfile.open((path+"/canvases.html").c_str());  
     indexfile << "<html>\n"
     << "<meta http-equiv=\"refresh\" content=\"10; URL=canvases.html\">\n"
     << "<body>"
     << "</body>\n</html>\n";
     indexfile.clear();
     indexfile.close();

     indexfile.open((path+"/imageout.html").c_str());
     indexfile << "<html>\n"
     << "<meta http-equiv=\"refresh\" content=\"10; URL=imageout.html\">\n"
     << "<body>"
     << "</body>\n</html>\n";
     indexfile.clear();
     indexfile.close(); 
  */
}

void CSCMonitor::createTreePage(std::string path)
{
  std::ofstream fout;
  fout.open((path+"/tree.html").c_str());
  fout << "<html>\n<head>\n"
       << "<title>Show Menu Items Tree</title>\n"
       << "<style>\n"
       << "a, A:link, a:visited, a:active, A:hover\n"
       << "        {color: #000000; text-decoration: none; font-family: Tahoma, Verdana; font-size: 12px}\n"
       << "</style>\n"
       << "</head>\n"
       << "<body>\n"

       << "<script language=\"JavaScript\" src=\"tree.js\"></script>\n"
       << "<script language=\"JavaScript\" src=\"tree_items.js\"></script>\n"
       << "<script language=\"JavaScript\" src=\"tree_tpl.js\"></script>\n"
       << "<table cellpadding=\"5\" cellspacing=\"0\" cellpadding=\"10\" border=\"0\" width=\"100%\">"
       << "<tr>\n"
       << "<td>\n"
       << "<script language=\"JavaScript\">\n"
       << "<!--//\n"
       << "        new tree (TREE_ITEMS, TREE_TPL);\n"
       << "//-->\n"
       << "</script>"
       << "</td>\n"
       << "</tr>\n"
       << "</table>\n"

       << "</body>\n"
       << "</html>" << endl;
  fout.close();

}



void CSCMonitor::createTreeEngine(std::string path ) 
{
  std::ofstream fout;
  fout.open((path+"/tree.js").c_str());

  fout  << "function tree(tm0,tm1)"
        << "{this.tm2=tm1;this.tm3=tm0;this.tm4=this;this.tm5=[];"
        << "this.tm6=null;this.tm7=-1;var tm8=new Image(),tm9=new Image();"
        << "tm8.src=tm1['icon_e'];tm9.src=tm1['icon_l'];tm1['im_e']=tm8;tm1['im_l']=tm9;"
        << "for(var i=0;i<64;i++)if(tm1['icon_'+i]){var tmA=new Image();tm1['im_'+i]=tmA;tmA.src=tm1['icon_'+i]}"
        << "this.tmB=function(tmC){var tmD=this.tm5[tmC];tmD.tmE(tmD.tmF)};"
        << "this.tmG=function(tmC){return this.tm5[tmC].tmG()};"
        << "this.tmH=function(tmC){this.tm5[tmC].tmI(true)};this.tmJ=function(tmC){this.tm5[tmC].tmI()};this.tmK=[];"
        << "for(var i=0;i<tm0.length;i++)new tmL(this,i);this.tmC=trees.length;trees[this.tmC]=this;for(var i=0;i<this.tmK.length;i++){document.write(this.tmK[i].tmM());this.tmK[i].tmE()}}function tmL(tmN,tmO){this.tm7=tmN.tm7+1;this.tm3=tmN.tm3[tmO+(this.tm7?2:0)];if(!this.tm3)return;this.tm4=tmN.tm4;this.tmN=tmN;this.tmO=tmO;this.tmF=!this.tm7;this.tmC=this.tm4.tm5.length;this.tm4.tm5[this.tmC]=this;tmN.tmK[tmO]=this;this.tmK=[];for(var i=0;i<this.tm3.length-2;i++)new tmL(this,i);this.tmP=tmQ;this.tmE=tmR;this.tmG=tmS;this.tmM=tmT;this.tmI=tmU;this.tmV=function(){return this.tmO==this.tmN.tmK.length-1}}function tmR(tmW){var tmX=tmY('i_div'+this.tm4.tmC+'_'+this.tmC);if(!tmX)return;if(!tmX.innerHTML){var tmK=[];for(var i=0;i<this.tmK.length;i++)tmK[i]=this.tmK[i].tmM();tmX.innerHTML=tmK.join('')}"
        << "tmX.style.display=(tmW?'none':'block');this.tmF=!tmW;var tmZ=document.images['j_img'+this.tm4.tmC+'_'+this.tmC],tma=document.images['i_img'+this.tm4.tmC+'_'+this.tmC];if(tmZ)tmZ.src=this.tmP(true);if(tma)tma.src=this.tmP();this.tmI()}function tmS(tmb){if(!tmb){var tmc=this.tm4.tm6;this.tm4.tm6=this;if(tmc)tmc.tmG(true)}var tma=document.images['i_img'+this.tm4.tmC+'_'+this.tmC];if(tma)tma.src=this.tmP();tmY('i_txt'+this.tm4.tmC+'_'+this.tmC).style.fontWeight=tmb?'normal':'bold';this.tmI();return Boolean(this.tm3[1])}"
        << "function tmU(tmd){window.setTimeout('window.status=\"'+(tmd?'':this.tm3[0]+(this.tm3[1]?' ('+this.tm3[1]+')':''))+'\"',10)}"
        << "function tmT(){var tme=[],tmf=this.tmN;for(var i=this.tm7;i>1;i--)"
        << "{tme[i]='<img src=\"'+this.tm4.tm2[tmf.tmV()?'icon_e':'icon_l']+'\" border=\"0\" align=\"absbottom\">';"
        << "tmf=tmf.tmN}return '<table cellpadding=\"0\" cellspacing=\"0\" border=\"0\"><tr><td nowrap>'+(this.tm7?tme.join('')+(this.tmK.length?'<a href=\"javascript: trees['+this.tm4.tmC+'].tmB('+this.tmC+')\" onmouseover=\"trees['+this.tm4.tmC+'].tmJ('+this.tmC+')\" onmouseout=\"trees['+this.tm4.tmC+'].tmH('+this.tmC+')\">"
        << "<img src=\"'+this.tmP(true)+'\" border=\"0\" align=\"absbottom\" name=\"j_img'+this.tm4.tmC+'_'+this.tmC+'\"></a>':'<img src=\"'+this.tmP(true)+'\" border=\"0\" align=\"absbottom\">'):'')+'<a href=\"'+this.tm3[1]+'\" target=\"'+this.tm4.tm2['target']+'\" onclick=\"return trees['+this.tm4.tmC+'].tmG('+this.tmC+')\" ondblclick=\"trees['+this.tm4.tmC+'].tmB('+this.tmC+')\" onmouseover=\"trees['+this.tm4.tmC+'].tmJ('+this.tmC+')\" onmouseout=\"trees['+this.tm4.tmC+'].tmH('+this.tmC+')\" class=\"t'+this.tm4.tmC+'i\" id=\"i_txt'+this.tm4.tmC+'_'+this.tmC+'\"><img src=\"'+this.tmP()+'\" border=\"0\" align=\"absbottom\" name=\"i_img'+this.tm4.tmC+'_'+this.tmC+'\" class=\"t'+this.tm4.tmC+'im\">'+this.tm3[0]+'</a></td></tr></table>'+(this.tmK.length?'<div id=\"i_div'+this.tm4.tmC+'_'+this.tmC+'\" style=\"display:none\"></div>':'')}function tmQ(tmg){return this.tm4.tm2['icon_'+((this.tm7?0:32)+(this.tmK.length?16:0)+(this.tmK.length&&this.tmF?8:0)+(!tmg&&this.tm4.tm6==this?4:0)+(tmg?2:0)+(tmg&&this.tmV()?1:0))]}var trees=[];tmY=document.all?function(tmh){return document.all[tmh]}:function(tmh){return document.getElementById(tmh)};" << endl;

  fout.close();

}



void CSCMonitor::createTreeTemplate(std::string path)
{
  std::ofstream fout;
  fout.open((path+"/tree_tpl.js").c_str());

  std::string treebase = "http://www.phys.ufl.edu/~barashko/dqm/";

  fout  << "var TREE_TPL = {\n"
        << "'target'  : 'frameset',\n" // name of the frame links will be opened in
    // other possible values are: _blank, _parent, _search, _self and _top

        << "'icon_e'  : '" << treebase << "icons/empty.gif',\n" // empty image
        << "'icon_l'  : '" << treebase << "icons/line.gif',\n"  // vertical line

        << "'icon_32' : '" << treebase << "icons/base.gif',\n"   // root leaf icon normal
        << "'icon_36' : '" << treebase << "icons/base.gif',\n"   // root leaf icon selected

        << "'icon_48' : '" << treebase << "icons/base.gif',\n"   // root icon normal
        << "'icon_52' : '" << treebase << "icons/base.gif',\n"   // root icon selected
        << "'icon_56' : '" << treebase << "icons/base.gif',\n"   // root icon opened
        << "'icon_60' : '" << treebase << "icons/base.gif',\n"   // root icon selected

        << "'icon_16' : '" << treebase << "icons/folder.gif',\n" // node icon normal
        << "'icon_20' : '" << treebase << "icons/folderopen.gif',\n" // node icon selected
        << "'icon_24' : '" << treebase << "icons/folderopen.gif',\n" // node icon opened
        << "'icon_28' : '" << treebase << "icons/folderopen.gif',\n" // node icon selected opened

        << "'icon_0'  : '" << treebase << "icons/page.gif',\n" // leaf icon normal
        << "'icon_4'  : '" << treebase << "icons/page.gif',\n" // leaf icon selected

        << "'icon_2'  : '" << treebase << "icons/joinbottom.gif',\n" // junction for leaf
        << "'icon_3'  : '" << treebase << "icons/join.gif',\n"       // junction for last leaf
        << "'icon_18' : '" << treebase << "icons/plusbottom.gif',\n" // junction for closed node
        << "'icon_19' : '" << treebase << "icons/plus.gif',\n"       // junctioin for last closed node
        << "'icon_26' : '" << treebase << "icons/minusbottom.gif',\n"// junction for opened node
        << "'icon_27' : '" << treebase << "icons/minus.gif'\n"      // junctioin for last opended node
        << "};" << endl;
  fout.close();
}


