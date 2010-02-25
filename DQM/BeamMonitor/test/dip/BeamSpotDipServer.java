//
//  File: BeamSpotDipServer.java   (W.Badgett, G.Y.Jeng)
//

package cms.dip.tracker.beamspot;

import cern.dip.*;
import java.lang.Thread;
import java.util.Random;
import java.io.*;

public class BeamSpotDipServer
extends Thread
implements Runnable,DipPublicationErrorHandler
{
  public final static String subjectCMS = "dip/CMS/Tracker/BeamSpot";
  public final static String subjectLHC = "dip/CMS/LHC/LuminousRegion";

  DipFactory dip;
  DipData messageCMS;
  DipData messageLHC;
  DipPublication publicationCMS;
  DipPublication publicationLHC;
  int runnum;
  String startTime;
  String endTime;
  String lumiRange;
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
      while (keepRunning)
      {
	try{
	    File myFile = new File("/nfshome0/jengbou/BeamFitResults.txt");
	    FileReader fr = new FileReader(myFile);
	    BufferedReader br = new BufferedReader(fr);
	    LineNumberReader lbr = new LineNumberReader(br);

	    long tmpTime = myFile.lastModified();
	    if (tmpTime > lastFitTime) {
		System.out.println("Read new record");
		lastFitTime = tmpTime;
	    } else {
		System.out.println("No new record");
		try { Thread.sleep(23000); }
		catch(InterruptedException e) {
		    keepRunning = false;
		}
		continue;
	    }
	    int recCount = 0;
	    int it = 0;
	    String record = new String();
	    System.out.println("Last line read = " + lastLine);
	    while ((record = br.readLine()) != null) {
		recCount++;
		if (lastLine >= recCount) {
		    continue;
		}
		it = recCount % 23;
		String[] tmp;
		tmp = record.split("\\s");
		switch(it) {
		case 1:
		    if (!record.startsWith("Run")){
			System.out.println("BeamFitResults text file may be corrupted. Stopping BeamSpot DIP Server!");
			System.exit(0);
		    }
 		    runnum = new Integer(tmp[1]);
		    System.out.println("Run: "+runnum);
		    break;
		case 2:
                    startTime = record.substring(15);
		    break;
		case 3:
                    endTime = record.substring(13);
		    break;
		case 4:
		    lumiRange = record.substring(10);
		    break;
		case 5:
		    type = new Integer(tmp[1]);
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

		default:
		    
		}
	    }
	    lastLine = recCount;
	} catch (Exception e) {
	    e.printStackTrace();
	}

	width_x = (float)Math.abs( random.nextGaussian() * 0.98 );
	width_y = (float)Math.abs( random.nextGaussian() * 1.20 );
	err_width_x = (float)Math.abs( random.nextGaussian() * 0.42 );
	err_width_y = (float)Math.abs( random.nextGaussian() * 0.56 );

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
	messageCMS.insert("type",type); //Unknown=-1, Fake=0, Tracker=2(Good)
	messageCMS.insert("x",x);
	messageCMS.insert("y",y);
	messageCMS.insert("z",z);
	messageCMS.insert("err_x",err_x);
	messageCMS.insert("err_y",err_y);
	messageCMS.insert("err_z",err_z);
	messageCMS.insert("dxdz",dxdz);
	messageCMS.insert("dydz",dydz);
	messageCMS.insert("err_dxdz",err_dxdz);
	messageCMS.insert("err_dydz",err_dydz);
	messageCMS.insert("width_x",width_x);
	messageCMS.insert("width_y",width_y);
	messageCMS.insert("sigma_z",sigma_z);
	messageCMS.insert("err_width_x",err_width_x);
	messageCMS.insert("err_width_y",err_width_y);
	messageCMS.insert("err_sigma_z",err_sigma_z);

	messageLHC.insert("Size",Size);
	messageLHC.insert("Centroid",Centroid);
	messageLHC.insert("Tilt",Tilt);

	DipTimestamp zeit = new DipTimestamp();
	publicationCMS.send(messageCMS, zeit);
	publicationLHC.send(messageLHC, zeit);
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
