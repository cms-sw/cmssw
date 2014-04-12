void DrawGraphs()
{
  std::cout<<"moving to the event: "<<++event<<std::endl;

  the_event = IntToString(event);
  can.Clear();
  can.Divide(width,height);
  can.Update();
  for(int k=0 ; k<width*height ; k++){
    std::string xtal = IntToString(windCry[k]);
    std::string name = "Graph_ev"+the_event+"_ic"+xtal;

    TGraph* gra = (TGraph*) f.Get(name.c_str()));
  int canvas_num = width*height - (k%height)*width - width + 1 + k/height;
  //cout<<name<<endl;
  can.cd(canvas_num);
  if( gra != NULL ){
    gra->GetXaxis()->SetTitle("sample");
    gra->GetYaxis()->SetTitle("adc");
    gra->Draw("A*");
    can.Update();
  }
  //else{gPad->Clear();}
}
can.cd((width*height+1)/2);
can.Update();
//return the_event;

}
