#include "DQM/CSCMonitorModule/interface/CSCMonitorObject.h"

std::map<int, std::string> ParseAxisLabels(std::string s)
{

  std::map<int, std::string> labels;
  std::string tmp = s;
  std::string::size_type pos = tmp.find("|");
  char* stopstring = NULL;

  while (pos != std::string::npos)
    {
      std::string label_pair = tmp.substr(0, pos);
      tmp.replace(0,pos+1,"");
      if (label_pair.find("=") != std::string::npos) {
	int nbin = strtol(label_pair.substr(0,label_pair.find("=")).c_str(),  &stopstring, 10);
	std:: string label = label_pair.substr(label_pair.find("=")+1, label_pair.length());
	while (label.find("\'") != std::string::npos) {
	  label.erase(label.find("\'"),1);
	}
	labels[nbin] = label;
      }
      pos = tmp.find("|");
    }
  return labels;
}

CSCMonitorObject::CSCMonitorObject(const CSCMonitorObject& mo)
{
  object = NULL;// reinterpret_cast<CSCMonitorElement*>(mo.object->Clone());
  type = mo.type;
  prefix = mo.prefix;
  folder = mo.folder;
  name = mo.name;
  title = mo.title;
  params = mo.params;
  QTest_result = mo.QTest_result;
}

CSCMonitorObject& CSCMonitorObject::operator=(const CSCMonitorObject& mo)
{
  object = NULL; //reinterpret_cast<CSCMonitorElement*>(mo.object->Clone());
  type = mo.type;
  prefix = mo.prefix;
  folder = mo.folder;
  name = mo.name;
  title = mo.title;
  params = mo.params;
  QTest_result = mo.QTest_result;
  return *this;
}

CSCMonitorObject::CSCMonitorObject() :
  type(""),
  prefix(""),
  name(""),
  title(""),
  folder("")
{
  object = NULL;
  QTest_result = 0;
  params.clear();
}

CSCMonitorObject::CSCMonitorObject(DOMNode* info)
{
  object = NULL;
  QTest_result = 0;
  parseDOMNode(info);
}


int CSCMonitorObject::Book(DaqMonitorBEInterface* dbe)
{

  int nbinsx = 0, nbinsy = 0, nbinsz = 0;
  double xlow = 0, ylow = 0, zlow = 0;
  double xup = 0, yup = 0, zup = 0;
  char *stopstring;
  /*
    if (object != NULL) {
    delete object;
    object = NULL;
    }
  */
  std::map<std::string, std::string> other_params;
  std::map<std::string, std::string>::iterator itr;
  if ((itr = params.find("XBins")) != params.end()) {
    nbinsx = strtol( itr->second.c_str(), &stopstring, 10 );
  }
  if ((itr = params.find("YBins")) != params.end()) {
    nbinsy = strtol( itr->second.c_str(), &stopstring, 10 );
  }
  if ((itr = params.find("ZBins")) != params.end()) {
    nbinsz = strtol( itr->second.c_str(), &stopstring, 10 );
  }
  if ((itr = params.find("XMin")) != params.end()) {
    xlow = strtol( itr->second.c_str(), &stopstring, 10 );
  }
  if ((itr = params.find("XMax")) != params.end()) {
    xup = strtol( itr->second.c_str(), &stopstring, 10 );
  }
  if ((itr = params.find("YMin")) != params.end()) {
    ylow = strtol( itr->second.c_str(), &stopstring, 10 );
  }
  if ((itr = params.find("YMax")) != params.end()) {
    yup = strtol( itr->second.c_str(), &stopstring, 10 );
  }
  if ((itr = params.find("ZMin")) != params.end()) {
    zlow = strtol( itr->second.c_str(), &stopstring, 10 );
  }
  if ((itr = params.find("ZMax")) != params.end()) {
    zup = strtol( itr->second.c_str(), &stopstring, 10 );
  }



  if ((itr = params.find("XRange")) != params.end()) {
    std::string str = itr->second;
    replace(str.begin(), str.end(), '(', ' ');
    replace(str.begin(), str.end(), ')', ' ');
    replace(str.begin(), str.end(), ':', ' ');
    std::stringstream st(str);
    st >> xlow;
    st >> xup;
  }
  if ((itr = params.find("YRange")) != params.end()) {
    std::string str = itr->second;
    replace(str.begin(), str.end(), '(', ' ');
    replace(str.begin(), str.end(), ')', ' ');
    replace(str.begin(), str.end(), ':', ' ');
    std::stringstream st(str);
    st >> ylow;
    st >> yup;
  }
  if ((itr = params.find("ZRange")) != params.end()) {
    std::string str = itr->second;
    replace(str.begin(), str.end(), '(', ' ');
    replace(str.begin(), str.end(), ')', ' ');
    replace(str.begin(), str.end(), ':', ' ');
    std::stringstream st(str);
    st >> zlow;
    st >> zup;
  }

  if (type.find("h1") != std::string::npos) {
    //	std::cout << getFullName() <<"("<< nbinsx<<","<<xlow<<","<<xup<<")"<<std::endl;
    //        object = new TH1F(getFullName().c_str(), getTitle().c_str(), nbinsx, xlow, xup);
    object = dbe->book1D(getFullName(), getTitle(), nbinsx, xlow, xup);
  } else
    if (type.find("h2") != std::string::npos) {
      // std::cout << getFullName() <<"("<< nbinsx<<","<<xlow<<","<<xup<<","<<nbinsy<<","<<ylow<<","<<yup<<")"<<std::endl;
      //        object = new TH2F(getFullName().c_str(), getTitle().c_str(), nbinsx, xlow, xup, nbinsy, ylow, yup);
      object = dbe->book2D(getFullName(), getTitle(), nbinsx, xlow, xup, nbinsy, ylow, yup);
    } else
      if (type.find("h3") != std::string::npos) {
	object = dbe->book3D(getFullName(), getTitle(), nbinsx, xlow, xup,
			     nbinsy, ylow, yup, nbinsz, zlow, zup);
	//        object = new TH3F(getFullName().c_str(), getTitle().c_str(), nbinsx, xlow, xup,
	//			nbinsy, ylow, yup, nbinsz, zlow, zup);
      }else
	if (type.find("hp2") != std::string::npos) {
//	  	object = dbe->bookProfile2D(getFullName(), getTitle(), nbinsx, xlow, xup,
//	  			nbinsy, ylow, yup, zlow, zup);
	  //        object = new TProfile2D(getFullName().c_str(), getTitle().c_str(), nbinsx, xlow, xup,
	  //			nbinsy, ylow, yup);
	} else
	  if (type.find("hp") != std::string::npos) {
//	    	object = dbe->bookProfile(getFullName(), getTitle(), nbinsx, xlow, xup, ylow, yup);
	    //         object = new TProfile(getFullName().c_str(), getTitle().c_str(), nbinsx, xlow, xup);
	  }


  // !!! TODO: Add object class check
  if (object != NULL) {
    // std::cout << "Booked " << getFullName() << std::endl;
    if (((itr = params.find("XTitle")) != params.end()) ||
	((itr = params.find("XLabel")) != params.end())) {
      // object->SetXTitle(itr->second.c_str());
      object->setAxisTitle(itr->second,1);
    }
    if (((itr = params.find("YTitle")) != params.end()) ||
	((itr = params.find("YLabel")) != params.end())) {
      object->setAxisTitle(itr->second,2);
      // object->SetYTitle(itr->second.c_str());
    }
    if (((itr = params.find("ZTitle")) != params.end()) ||
	((itr = params.find("ZLabel")) != params.end())) {
      object->setAxisTitle(itr->second,3);
      // object->SetZTitle(itr->second.c_str());
    }

    if ((itr = params.find("SetOption")) != params.end()) {
      MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
      if (ob) {
	if (type.find("h1") != std::string::npos) {
		TH1F* root_ob = dynamic_cast<TH1F*> ( ob->operator->() );
		if (root_ob)
			root_ob->SetOption(itr->second.c_str());
	} else 	if (type.find("h2") != std::string::npos) {
		TH2F* root_ob = dynamic_cast<TH2F*> ( ob->operator->() );
                if (root_ob)
                        root_ob->SetOption(itr->second.c_str());
	}

      }
      //     object->SetOption(itr->second.c_str());

    }

    if ((itr = params.find("SetOptStat")) != params.end()) {
      gStyle->SetOptStat(itr->second.c_str());

    }

    if ((itr = params.find("SetStats")) != params.end()) {
      int stats = strtol( itr->second.c_str(), &stopstring, 10 );
      MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
      if (ob) {
	( (TH1F *) ob->operator->() )->SetStats(bool(stats));
      }

      //object->SetStats(bool(stats));

    }


    if ((itr = params.find("SetFillColor")) != params.end()) {
      int color = strtol( itr->second.c_str(), &stopstring, 10 );
      MonitorElementT<TH1> * ob = dynamic_cast<MonitorElementT<TH1>*>(object);
      if (ob) {
	if (type.find("h1") != std::string::npos) {
                TH1F* root_ob = dynamic_cast<TH1F*> ( ob->operator->() );
                if (root_ob)
                        root_ob->SetFillColor(color);
        } else  if (type.find("h2") != std::string::npos) {
                TH2F* root_ob = dynamic_cast<TH2F*> ( ob->operator->() );
                if (root_ob)
                        root_ob->SetFillColor(color);
        }

//	( (TH1F *) ob->operator->() )->SetFillColor(color);
//	ob->operator->()->SetFillColor(color);
	
      }

      //     object->SetFillColor(color);
    }
    if ((itr = params.find("SetXLabels")) != params.end()) {
      std::map<int, std::string> labels = ParseAxisLabels(itr->second);
      for (std::map<int, std::string>::iterator l_itr = labels.begin(); l_itr != labels.end(); ++l_itr)
	{
	  object->setBinLabel(l_itr->first, l_itr->second,1);
	}

    }

    if ((itr = params.find("SetYLabels")) != params.end()) {
      std::map<int, std::string> labels = ParseAxisLabels(itr->second);
      for (std::map<int, std::string>::iterator l_itr = labels.begin(); l_itr != labels.end(); ++l_itr)
	{
	  object->setBinLabel(l_itr->first, l_itr->second,2);
	}
    }
    if ((itr = params.find("LabelsOption")) != params.end()) {
      std::string st = itr->second;
      if (st.find(",") != std::string::npos) {
	std::string opt = st.substr(0,st.find(",")) ;
	std::string axis = st.substr(st.find(",")+1,st.length());
	MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
	if (ob) {
	  ( (TH1 *) ob->operator->() )->LabelsOption(opt.c_str(),axis.c_str());
	}
      }
    }
    if ((itr = params.find("SetLabelSize")) != params.end()) {
      std::string st = itr->second;
      if (st.find(",") != std::string::npos) {
	double opt = atof(st.substr(0,st.find(",")).c_str()) ;
	std::string axis = st.substr(st.find(",")+1,st.length());
	MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
	if (ob) {
	  ( (TH1 *) ob->operator->() )->SetLabelSize(opt,axis.c_str());
	}
      }
    }
    if ((itr = params.find("SetTitleOffset")) != params.end()) {
      std::string st = itr->second;
      if (st.find(",") != std::string::npos) {
	double opt = atof(st.substr(0,st.find(",")).c_str()) ;
	std::string axis = st.substr(st.find(",")+1,st.length());
	MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
	if (ob) {
	  ( (TH1 *) ob->operator->() )->SetTitleOffset(opt,axis.c_str());
	}
      }
    }

    if ((itr = params.find("SetNdivisionsX")) != params.end()) {
      int opt = strtol( itr->second.c_str(), &stopstring, 10 );
      MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
      if (ob) {
        ( (TH1 *) ob->operator->() )->SetNdivisions(opt,"X");
      }
    }

    if ((itr = params.find("SetNdivisionsY")) != params.end()) {
      int opt = strtol( itr->second.c_str(), &stopstring, 10 );
      MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
      if (ob) {
        ( (TH1 *) ob->operator->() )->SetNdivisions(opt,"Y");
      }
    }

    if ((itr = params.find("SetTickLengthX")) != params.end()) {
      std::string st = itr->second;
      double opt = atof(st.c_str()) ;
      MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
      if (ob) {
        ( (TH1 *) ob->operator->() )->SetTickLength(opt,"X");
      }
    }
    
    if ((itr = params.find("SetTickLengthY")) != params.end()) {
      std::string st = itr->second;
      double opt = atof(st.c_str()) ;
      MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
      if (ob) {
        ( (TH1 *) ob->operator->() )->SetTickLength(opt,"Y");
      }
    }

    if ((itr = params.find("SetLabelSizeX")) != params.end()) {
      std::string st = itr->second;
      double opt = atof(st.c_str()) ;  
      MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
      if (ob) {
        ( (TH1 *) ob->operator->() )->GetXaxis()->SetLabelSize(opt);
      }
    }

    if ((itr = params.find("SetLabelSizeY")) != params.end()) {
      std::string st = itr->second;
      double opt = atof(st.c_str()) ;
      MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
      if (ob) {
        ( (TH1 *) ob->operator->() )->GetYaxis()->SetLabelSize(opt);
      }
    }

    MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
    if (ob) {
	( (TH1F *) ob->operator->() )->SetFillColor(DEF_HISTO_COLOR);
    }

  }

  return 0;
}



int CSCMonitorObject::Book(DOMNode* info, DaqMonitorBEInterface* dbe)
{

  parseDOMNode(info);
  Book(dbe);
  return 0;
}

CSCMonitorObject::~CSCMonitorObject()
{
  /*
    if (object != NULL) {
    delete object;
    object = NULL;
    }
  */
}

void CSCMonitorObject::Draw()
{
  MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
  if (ob) {
    ob->operator->()->Draw();
  }
}

void CSCMonitorObject::Write()
{
  MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
  if (ob) {
    ob->operator->()->Write();
  }
}

void CSCMonitorObject::Reset()
{
  /*
  MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
  if (ob) {
   //  ob->operator->()->Reset();
  }
  */
}
void CSCMonitorObject::setName(std::string newname)
{
  name = newname;
  if (object != NULL) {
    // object->SetName(getFullName().c_str());
  }
}

void CSCMonitorObject::setPrefix(std::string newprefix)
{
  prefix = newprefix;
  if (object != NULL) {
    // object->SetName(getFullName().c_str());
  }
}


void CSCMonitorObject::setTitle(std::string newtitle)
{
  title = newtitle;
  if (object != NULL) {
    // object->SetTitle(getTitle().c_str());
    object->setTitle(getTitle());
  }
}

int CSCMonitorObject::setParameter(std::string parname, std::string parvalue)
{
  if (object != NULL) {
    params[parname] = parvalue;
    return 0;
  } else return 1;
	
}

void CSCMonitorObject::SetEntries(double entries)
{
  if (object != NULL) {
    // object->SetEntries(entries);
    object->setEntries(entries);
  }

}

void CSCMonitorObject::SetBinContent(int nbin, double value)
{
  if (object != NULL) {
    // object->SetBinContent(nbin, value);
    object->setBinContent(nbin, value);
  }

}

void CSCMonitorObject::SetBinContent(int nxbin, int nybin, double value)
{
  if (object != NULL) {
    // object->SetBinContent(nxbin,nybin,value);
    object->setBinContent(nxbin,nybin,value);
  }

}

void CSCMonitorObject::SetNormFactor(double value)
{
  if (object != NULL) {
    MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
    if (ob) {
     ( (TH1F *) ob->operator->() )->SetNormFactor(value);
//      ob->operator->()->SetNormFactor(value);
    }

    // object->SetNormFactor(value);
  }

}

void CSCMonitorObject::SetBinError(int binx, double error)
{
  if (object != NULL) {
    // object->SetBinError(binx, error);
    object->setBinError(binx, error);
  }

}

double CSCMonitorObject::GetBinError(int nbin)
{
  if (object != NULL) {
    // return object->GetBinError(nbin);
    return object->getBinError(nbin);
  } else return 0;

}

double CSCMonitorObject::GetBinContent(int nbin)
{
  if (object != NULL) {
    // return object->GetBinContent(nbin);
    return object->getBinContent(nbin);
  } else return 0;

}



double CSCMonitorObject::GetBinContent(int nxbin, int nybin)
{
  if (object != NULL) {
    // return object->GetBinContent(nxbin, nybin);
    return object->getBinContent(nxbin, nybin);
  } else return 0;

}

int CSCMonitorObject::GetMaximumBin()
{
  if (object != NULL) {
    MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
    if (ob) {
      return ( (TH1F *) ob->operator->() )->GetMaximumBin();
      // return ob->operator->()->GetMaximumBin();
    }

    // return object->GetMaximumBin();
    return 0;
  } else return 0;
}

double CSCMonitorObject::GetEntries()
{
  if (object != NULL) {
    // return object->GetEntries();

    return object->getEntries();
  } else return 0;
}

void CSCMonitorObject::SetAxisRange(double xmin, double xmax, std::string options)
{
  if (object != NULL) {
    MonitorElementT<TNamed> * ob = dynamic_cast<MonitorElementT<TNamed>*>(object);
    if (ob) {
      ( (TH1F *) ob->operator->() )->SetAxisRange(xmin, xmax, options.c_str());
//      ob->operator->()->SetAxisRange(xmin, xmax, options.c_str());
    }
  }

}

void CSCMonitorObject::SetAxisRange(double xmin, double xmax, int axis=1)
{
  if (object != NULL) {
    object->setAxisRange(xmin, xmax, axis);
  }

}


void CSCMonitorObject::SetBinLabel(int nbin, std::string label, int axis=1)
{
  if (object != NULL) {		
    object->setBinLabel(nbin, label, axis);
  }

}


int CSCMonitorObject::setParameters(std::map<std::string, std::string> newparams, bool resetParams)
{
  std::map<std::string, std::string>::iterator itr;
  if (resetParams) {
    params.clear();
    params = newparams;
  } else {
    // == Append to parameters list
    for (itr = newparams.begin(); itr != newparams.end(); ++itr) {
      params[itr->first] = itr->second;
    }
  }
  return 0;
}

std::string CSCMonitorObject::getParameter(std::string paramname) 
{
  std::map<std::string, std::string>::iterator itr;
  if ((itr = params.find(paramname)) != params.end()) 
    return itr->second;
  else
    return "";
}

int CSCMonitorObject::Fill(double xval)
{
  if (object != NULL) {
    // std::cout << name << ":" << xval << std::endl;
    // return object->Fill(xval);
    object->Fill(xval);
    return 0;
  } else return 1;

}

int CSCMonitorObject::Fill(double xval, double yval)
{
  int res = 1;
  if (object != NULL) {
    //		if (std::string(object->ClassName()).find("TH2") != std::string::npos)
    // std::cout << name << ":" << xval << ":" << yval << std::endl;
    // return object->Fill(xval, yval);
    object->Fill(xval, yval);
    return 0;
  }
  return res;

}

int CSCMonitorObject::Fill(double xval, double yval, double zval)
{
  int res = 1;
  if (object != NULL) {
    // if (std::string(object->ClassName()).find("TH3") != std::string::npos)
    //   return reinterpret_cast<TH3*>(object)->Fill(xval, yval, zval);
    object->Fill(xval, yval, zval);
    return 0;
  }
  return res;
}

int CSCMonitorObject::doQTest()
{
  QTest_result = 1;
  return QTest_result;
}

int CSCMonitorObject::Fill(double xval, double yval, double zval, double wval)
{
  int res = 1;
  if (object != NULL) {
    // if (std::string(object->ClassName()).find("TH3") != std::string::npos)
    // return reinterpret_cast<TH3*>(object)->Fill(xval, yval, zval, wval);
    object->Fill(xval, yval, zval, wval);
    return 0;
  }
  return res;

}


int CSCMonitorObject::parseDOMNode(DOMNode* info)
{
  std::map<std::string, std::string> obj_info;
  std::map<std::string, std::string>::iterator itr;
  DOMNodeList *children = info->getChildNodes();
  for(unsigned int i=0; i<children->getLength(); i++){
    std::string paramname = std::string(XMLString::transcode(children->item(i)->getNodeName()));
    if ( children->item(i)->hasChildNodes() ) {
      std::string param = std::string(XMLString::transcode(children->item(i)->getFirstChild()->getNodeValue()));
      obj_info[paramname] = param;
    }
  }
  /*
    for  (itr = obj_info.begin(); itr != obj_info.end(); ++itr) {
    std::cout << itr->first << ":" << itr->second << std::endl;
    }
  */

  if (obj_info.size() > 0) {
    // == Construct Monitoring Object Name
    if ((itr = obj_info.find("Type")) != obj_info.end()) {
      type = itr->second;
      //	obj_info.erase("Type");
		
    }
    if ((itr = obj_info.find("Prefix")) != obj_info.end()) {
      prefix = itr->second;
      //	obj_info.erase("Prefix");		
    }
    if ((itr = obj_info.find("Name")) != obj_info.end()) {
      name = itr->second;
      //	obj_info.erase("Name");
    }
	  
    // == Get Monitoring Object Title
    if ((itr = obj_info.find("Title")) != obj_info.end()) {
      title = itr->second;
      //	obj_info.erase("Title");
    }
    if ((itr = obj_info.find("Folder")) != obj_info.end()) {
      folder = itr->second;
      //      obj_info.erase("Folder");
    }


    // == Create Monitoring Object Parameters map
    params.clear();
    for (itr = obj_info.begin(); itr != obj_info.end(); ++itr) {
      params[itr->first] = itr->second; 
    }
  }
  return 0;
}
