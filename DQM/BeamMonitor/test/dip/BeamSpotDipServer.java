//
//  File: BeamSpotDipServer.java   (W.Badgett, G.Y.Jeng)
//

package cms.dip.tracker.beamspot;

import cern.dip.*;
import java.lang.Thread;
import java.util.Random;
import java.io.*;
import java.text.*;
import java.util.Date;

public class BeamSpotDipServer
extends Thread
implements Runnable,DipPublicationErrorHandler
{
  public final static String subjectCMS = "dip/CMS/Tracker/BeamSpot";
  public final static String subjectLHC = "dip/CMS/LHC/LuminousRegion";
  public final static String sourceFile = "/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt";
  public final static int lnPerRcd = 23;
  public final static int msPerLS = 23000; // ms
  public static boolean overwriteFlag = true; //if true, flag with flags[0]
  public static boolean publishStatErrors = false;

  private String[] flags = {"UNCERTAIN","BAD","GOOD"};

  DipFactory dip;
  DipData messageCMS;
  DipData messageLHC;
  DipPublication publicationCMS;
  DipPublication publicationLHC;
  int runnum;
  String startTime;
  String endTime;
  String lumiRange;
  String flag;
  int type;
  float x;
  float y;
  float z;
  float dxdz;
  float dydz;
  float err_x;
  float err_y;
  float err_z;
  float err_dxdz;
  float err_dydz;
  float width_x;
  float width_y;
  float sigma_z;
  float err_width_x;
  float err_width_y;
  float err_sigma_z;
  float Size[] = new float[3];
  float Centroid[] = new float[3];
  float Tilt[] = new float[2];

  boolean keepRunning;
  Random random = new Random((long)0xadeadcdf);
  long lastFitTime = 0;
  int lastLine = 0;

  public void handleException(DipPublication publication,
			      DipException e)
  {
    System.out.println("Error handler for " + 
		       publication.getTopicName() + 
		       " called because " + e.getMessage());
    e.printStackTrace();
  }

  public void run()
  {
    java.util.Date now = new java.util.Date();

    try
    {
      dip = Dip.create("CmsBeamSpot_"+now.getTime());

      System.out.println("Making publication " + subjectCMS);
      publicationCMS = dip.createDipPublication(subjectCMS, this);
      messageCMS = dip.createDipData();

      System.out.println("Making publication " + subjectLHC);
      publicationLHC = dip.createDipPublication(subjectLHC, this);
      messageLHC = dip.createDipData();

      keepRunning = true;
    }
    catch ( DipException e )
    {
      keepRunning = false;
    }
    try
    {
      int lsCount = 0;
      flag = flags[0];
      while (keepRunning)
      {
	try{
	    File myFile = new File(sourceFile);
	    myFile.createNewFile();
	    FileReader fr = new FileReader(myFile);
	    BufferedReader br = new BufferedReader(fr);
	    LineNumberReader lbr = new LineNumberReader(br);

	    long tmpTime = myFile.lastModified();
	    if ((lastFitTime != 0) && (tmpTime > lastFitTime)){
		if (myFile.length() > 0) {
		    System.out.println("Read new record");
		    lastFitTime = tmpTime;
		}
		else {
		    System.out.println("New Run Started");
		    lastFitTime = tmpTime;
		    lastLine = 0;
		    lsCount = 0;
		    continue;
		}
	    }
	    else {
		if (lastFitTime == 0) {//executed when server starts
		    int countln = 0;
		    while (lbr.readLine() != null){
			countln++;
		    }
		    // read the last record if server restatred during a run
		    lastLine = countln - lnPerRcd;}
		if (lsCount%10 == 0) {
		    System.out.println("Waiting for data...");
		    lastFitTime = tmpTime;
		}
		lsCount++;
		try { Thread.sleep(msPerLS); }
		catch(InterruptedException e) {
		    keepRunning = false;
		}
		continue;
	    }
	    lsCount = 0;
	    int nthLnInFile = 0;
	    int nthLnInRcd = 0;
	    String record = new String();
	    //System.out.println("Last line read = " + lastLine);
	    while ((record = br.readLine()) != null) {
		nthLnInFile++;
		if (lastLine >= nthLnInFile) {
		    continue;
		}
		nthLnInRcd = nthLnInFile % lnPerRcd;
		String[] tmp;
		tmp = record.split("\\s");
		switch(nthLnInRcd) {
		case 1:
		    if (!record.startsWith("Run")){
			System.out.println("BeamFitResults text file may be corrupted. Stopping BeamSpot DIP Server!");
			System.exit(0);
		    }
 		    runnum = new Integer(tmp[1]);
		    System.out.println("Run: " + runnum);
		    break;
		case 2:
                    startTime = record.substring(15);
		    break;
		case 3:
                    endTime = record.substring(13);
		    System.out.println("Time of fit: " + endTime);
		    break;
		case 4:
		    lumiRange = record.substring(10);
		    System.out.println("LS: " + lumiRange);
		    break;
		case 5:
		    type = new Integer(tmp[1]);
		    if (overwriteFlag) {
			flag = flags[0];
		    }
		    else if (type >= 2) flag = flags[2];
		    else flag = flags[1];
		    break;
		case 6:
		    x = new Float(tmp[1]);
		    System.out.println("x0      = " + x);
		    break;
		case 7:
		    y = new Float(tmp[1]);
		    System.out.println("y0      = " + y);
		    break;
		case 8:
		    z = new Float(tmp[1]);
		    System.out.println("z0      = " + z);
		    break;
		case 9:
		    sigma_z = new Float(tmp[1]);
		    System.out.println("sigma_z = " + sigma_z);
		    break;
		case 10:
		    dxdz = new Float(tmp[1]);
		    break;
		case 11:
		    dydz = new Float(tmp[1]);
		    break;
		case 12:
		    width_x = new Float(tmp[1]);
		    break;
		case 13:
		    width_y = new Float(tmp[1]);
		    break;
		case 14:
		    err_x = new Float(Math.sqrt(Double.parseDouble(tmp[1])));
		    //System.out.println(err_x);
		    break;
		case 15:
		    err_y = new Float(Math.sqrt(Double.parseDouble(tmp[2])));
		    //System.out.println(err_y);
		    break;
		case 16:
		    err_z = new Float(Math.sqrt(Double.parseDouble(tmp[3])));
		    //System.out.println(err_z);
		    break;
		case 17:
		    err_sigma_z = new Float(Math.sqrt(Double.parseDouble(tmp[4])));
		    //System.out.println(err_sigma_z);
		    break;
		case 18:
		    err_dxdz = new Float(Math.sqrt(Double.parseDouble(tmp[5])));
		    //System.out.println(err_dxdz);
		    break;
		case 19:
		    err_dydz = new Float(Math.sqrt(Double.parseDouble(tmp[6])));
		    //System.out.println(err_dydz);
		    break;
		case 20:
		    err_width_x = new Float(Math.sqrt(Double.parseDouble(tmp[7])));
		    err_width_y = err_width_x;
		    break;

		default:
		    break;
		}
	    }
	    lastLine = nthLnInFile;
	} catch (Exception e) {
	    e.printStackTrace();
	}

	Centroid[0] = x;
	Centroid[1] = y;
	Centroid[2] = z;

	Size[0] = width_x;
	Size[1] = width_y;
	Size[2] = sigma_z;

	Tilt[0] = dxdz;
	Tilt[1] = dydz;

	messageCMS.insert("runnum",runnum);
	messageCMS.insert("startTime",startTime);
	messageCMS.insert("endTime",endTime);
	messageCMS.insert("lumiRange",lumiRange);
	messageCMS.insert("flag",flag);
	messageCMS.insert("type",type); //Unknown=-1, Fake=0, Tracker=2(Good)
	messageCMS.insert("x",x);
	messageCMS.insert("y",y);
	messageCMS.insert("z",z);
	messageCMS.insert("dxdz",dxdz);
	messageCMS.insert("dydz",dydz);
	messageCMS.insert("width_x",width_x);
	messageCMS.insert("width_y",width_y);
	messageCMS.insert("sigma_z",sigma_z);
	if (publishStatErrors) {
	    messageCMS.insert("err_x",err_x);
	    messageCMS.insert("err_y",err_y);
	    messageCMS.insert("err_z",err_z);
	    messageCMS.insert("err_dxdz",err_dxdz);
	    messageCMS.insert("err_dydz",err_dydz);
	    messageCMS.insert("err_width_x",err_width_x);
	    messageCMS.insert("err_width_y",err_width_y);
	    messageCMS.insert("err_sigma_z",err_sigma_z);
	}
	messageLHC.insert("Size",Size);
	messageLHC.insert("Centroid",Centroid);
	messageLHC.insert("Tilt",Tilt);

	try {
	    long epoch = new SimpleDateFormat("yyyy.MM.dd HH:mm:ss zz").parse(endTime).getTime();
	    //System.out.println(epoch);
	    DipTimestamp zeit = new DipTimestamp(epoch);
	    publicationCMS.send(messageCMS, zeit);
	    publicationLHC.send(messageLHC, zeit);
	}
	catch (ParseException e) {
	    System.out.println("Publishing failed due to time parsing because " + e.getMessage());
	    e.printStackTrace();
	}
      }
    }
    catch (DipException e)
    {
      System.out.println("Failed to send data because " + e.getMessage());
      e.printStackTrace();
    }
  }
    
    
  public static void main(String args[])
  {
    BeamSpotDipServer server = new BeamSpotDipServer();
    server.start();
  }
}
