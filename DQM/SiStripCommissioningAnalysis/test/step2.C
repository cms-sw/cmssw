void step2(int runnumber, int ishalow, int ishahigh, int ishastep, int vfslow, int vfshigh, int vfsstep)
{
  gRoot->LoadMacro("CalibrationScanAnalysis.C+");
  CalibrationScanAnalysis analysis;
  {
    for(int isha = ishalow; isha<ishahigh; isha+=ishastep)
      for(int vfs = vfslow; vfs<vfshigh; vfs+=vfsstep)
        analysis.addFile(Form("SiStripCommissioningClient_000%d_ISHA%d_VFS%d.root",runnumber,isha,vfs));
  }
  analysis.analyze();
  analysis.draw();
  analysis.print();
  //analysis.sanitize(); // not mandatory... to be decided by an expert
  analysis.save("ishavfsScan.txt");
}
