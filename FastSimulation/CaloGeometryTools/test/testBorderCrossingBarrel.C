{
  // You should run "cmsRun testCaloGeometryTools.cfg" first
  TH2F* frame = new TH2F("h2","h2",100,-1.6,1.6,100,-3.2,3.2);
  frame->Draw();
  TFile * = new TFile("Grid.root");
  TIter nextkey( gDirectory->GetListOfKeys() );
  TKey *key, *oldkey=0;;
  unsigned cont=0;
  while ( (key = (TKey*)nextkey()))
    {
      
      //keep only the highest cycle number for each key
      if (oldkey && !strcmp(oldkey->GetName(),key->GetName()) || !TString(key->GetName()).BeginsWith("iBCB")) continue;
      TObject *obj = key->ReadObj();
      if ( obj->IsA()==TMarker().Class()  ) 
	{
	  //	  cout << key->GetName() << endl;
	  //	  (TText*)obj->Print() ;
	  TMarker * myMarker=(TMarker*)obj->Clone();
	  myMarker->Draw();
	  ++cont;
	}
      oldkey=key;
    }
} 
