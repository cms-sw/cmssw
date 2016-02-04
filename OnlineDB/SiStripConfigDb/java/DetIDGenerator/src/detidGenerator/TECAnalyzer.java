package detidGenerator;

//import db.*;
import fr.in2p3.ipnl.db.*;
import java.util.ArrayList;

/**
 * <p>Class used to get the DET IDs of the modules in the TEC</p>
 * @author G. Baulieu
 * @version 1.0
 */

/*
  $Date: 2007/03/22 20:46:49 $
  
  $Log: TECAnalyzer.java,v $
  Revision 1.3  2007/03/22 20:46:49  gbaulieu
  New numbering of Det ID

  Revision 1.2  2007/01/18 17:04:45  gbaulieu
  Use an external library for database connections
  Use preparedStatements to speed up the queries

  Revision 1.1  2006/06/28 11:42:24  gbaulieu
  First import of the sources

  Revision 1.9  2006/06/13 09:46:52  baulieu
  Correct the shift in the petal numbering for TEC- (ie Sector 8 contains petals F1 & B8)

  Revision 1.8  2006/05/10 07:47:51  baulieu
  Correct a bug in the det_id when there are several sectors with petal on a disk.
  Change the way the length of the fibers are found.
  Allow to print the result on the screen instead of exporting to a db.
  3 possibles parameters : -mtcc, -export, -help.
  Some documentation.

  Revision 1.7  2006/05/05 15:01:56  baulieu
  Correct the stereo value:
  0 normal mono
  1 stereo
  2 glued mono

  Revision 1.6  2006/03/27 08:12:13  baulieu
  Add an option for the MTCC detectors

  Revision 1.5  2006/03/21 17:04:13  baulieu
  New version of the TEC det_id (no fw/bw modules)

  Revision 1.4  2006/02/02 17:17:00  baulieu
  Some modifications for JDK 1.5
  Call a PL/SQL function to export the parameters


*/

public class TECAnalyzer implements IDetIdGenerator{

    private ArrayList<ArrayList<String>> detIds;
    private CDBConnection c;

    /**
       Default constructor
    */
    public TECAnalyzer() throws java.sql.SQLException, java.lang.ClassNotFoundException{
	detIds = new ArrayList<ArrayList<String>>();
	c = CDBConnection.getConnection();
	
	if(!DetIDGenerator.mtcc){//static member of DetIDGenerator : is it for the magnet test?
	    String detID = "1.6.1";
	    String tecVersion = "-";
	    
	    // for both TECs
	    for(int i=0;i<2;i++){
		ArrayList<ArrayList<String>> v = c.selectQuery("select object_id from cmstrkdb.object_assembly where object='TEC' and version='"+tecVersion+"'");
		if(v.size()==1){
		    String tecID = (String)((ArrayList)v.get(0)).get(0);
		    getWheels(tecID, detID);
		}
		
		tecVersion = "+";
		detID = detID.substring(0,3)+".2";
	    }
	}
	else{ // Petals for the Magnet test : 30250400000071 (back) and 30250200000034 (front)
	    //front
	    String detID = "1.6.2.1.1.1";
	    getModules("30250200000034", detID);
	    //back
	    detID = "1.6.2.1.0.1";
	    getModules("30250400000071", detID);
	}
    }

    /**
       Get the det IDs of the modules in the given TEC
       @param TECID The object_id of the TEC
       @param detID The current detID
    **/
    private void getWheels(String TECID, String detID) throws java.sql.SQLException{
	ArrayList<ArrayList<String>> v = c.selectQuery("select object_id, number_in_container from cmstrkdb.object_assembly "+
				 "where object='DISK' and container_id='"+
				 TECID+"' order by number_in_container");
	// for each DISK in the TEC
	for(int i=0;i<v.size();i++){
	    String wheelID = (v.get(i)).get(0);
	    String wheelNum = (v.get(i)).get(1);
	    String d = detID+"."+wheelNum;
	    getPetals(wheelID, d, true);
	    getPetals(wheelID, d, false);
	}
    }

    /**
       Get the det IDs of the modules in the given disk
       @param DISKID The object_id of the disk
       @param detID The current detID
       @param forward True for the front petal, false for the back
    **/
    private void getPetals(String DISKID, String detID, boolean forward) throws java.sql.SQLException{
	ArrayList<ArrayList<String>> v = c.selectQuery("select object_id, number_in_container from cmstrkdb.object_assembly "+
						 "where object='TECCR' and container_id='"+
						 DISKID+"' order by number_in_container");
	// for each control ring in the disk
	for(int i=0;i<v.size();i++){
	    String CRID = (v.get(i)).get(0);
	    int sector = Integer.parseInt((v.get(i)).get(1));
	    int tec = Integer.parseInt(detID.substring(4,5));

	    ArrayList<ArrayList<String>> v2 = c.selectQuery("select object_id, number_in_container from "+
						      "cmstrkdb.object_assembly "+
						      "where object='PETAL' and container_id='"+
						      CRID+"' and number_in_container="+(forward?"1":"2"));
	    //Get the front or back petal
	    for(int j=0;j<v2.size();j++){
		String nDetID = detID+(forward?".2.":".1.");
		if(tec!=1 || (tec==1 && !forward))
		    nDetID += sector;
		else//TEC- forward : shift in the petal numbering (Sector 1 contains B1 and F2)
		    nDetID += (sector%8)+1;
		
		String petalID = (v2.get(0)).get(0);
		getModules(petalID, nDetID);
	    }
	}
    }

    /**
       Get the det IDs of the modules in the given petal
       @param petalID The object_id of the petal
       @param detID The current detID
    **/
    private void getModules(String petalID, String detID) throws java.sql.SQLException{
	ArrayList<ArrayList<String>> v = c.selectQuery("select OA.object_id, OA.number_in_container, OD.type_description "+
						 "from cmstrkdb.object_assembly OA, "+
						 "cmstrkdb.object_description OD "+
						 "where OA.object=OD.object AND OA.type=OD.type "+
						 "AND OA.version=OD.version "+
						 "AND OA.object='MOD' and OA.container_id='"+
						 petalID+"' order by OA.number_in_container");
	// for each module in the petal
	for(int i=0;i<v.size();i++){
	    //the first digit of the position is the ring, the second digit the position in the ring
	    String modID = (v.get(i)).get(0);
	    String modPos = (v.get(i)).get(1);
	    String modType = (v.get(i)).get(2);
	    int wheel = Integer.parseInt(detID.substring(6,7));
	    int petalType = Integer.parseInt(detID.substring(8,9));//0 forward, 1 backward
	    int ring = Integer.parseInt(modPos.substring(0,1));
	    int pos =  Integer.parseInt(modPos.substring(1,2));
	    String stereo;
	    /*
	      If we are on rings 1,2 or 5 we have stereo modules
	      We count only one position for 2 modules (1 mono, the other stereo)
	      The distinction will be made with the stereo tag
	    */
	    if(ring==1 || ring==2 || ring==5){
		pos = (int)Math.ceil(((double)pos)/2);
		stereo = "2";//glued module if not stereo
	    }
	    else{
		stereo = "0";//normal module if not stereo
	    }
	    
	    //mono or stereo?
	    if(modType.indexOf("S.")!=-1)
		stereo = "1";// stereo module

	    /*
	      Correct the ring number :
	      7 rings for disks 1 to 3
	      6 rings for disks 4 to 6
	      5 rings for disks 7 to 8
	      4 rings for disk 9
	      one ring to rule them all! ... sorry :-)

	      We always start counting from 1 : 28/04/06 this should not be done any more
	    */
	    /*
	      if(wheel>3 && wheel<7)
	      ring -= 1;
	      if(wheel>6 && wheel<9)
	      ring -= 2;
	      if(wheel==9)
	      ring -= 3;
	      if(DetIDGenerator.mtcc)//it's a petal for disk 9 (only 4 rings) but on a disk 1 ...:-/
	      ring -= 3;
	    */
	    String d = detID+"."+ring+"."+pos;

	    d+="."+stereo;

	    // the detID is complete -> add it to the list
	    ArrayList<String> couple = new ArrayList<String>();
	    couple.add(modID);
	    couple.add(d);
	    
	    detIds.add(couple);
	}
    }

    public ArrayList<ArrayList<String>> getDetIds(){
	return detIds;
    }

}
