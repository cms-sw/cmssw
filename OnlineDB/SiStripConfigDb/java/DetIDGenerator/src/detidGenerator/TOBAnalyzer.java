package detidGenerator;

//import db.*;
import fr.in2p3.ipnl.db.*;
import java.util.ArrayList;
import java.util.Hashtable;

import java.net.URL;

import java.io.File;
import java.io.FileReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.BufferedReader;

public class TOBAnalyzer implements IDetIdGenerator{

    private final String ROD_POSITION_FILE = "/ressources/RodNumbering.csv";
    private final String TOB_PREFIXE = "1.5";

    private ArrayList<ArrayList<String>> detIds;
    private CDBConnection c;
    private Hashtable<String, Integer> rodPositionMap;

    /**
       Constructor.
       Loads the file containing the Rod numbering rules in the Hashtable,
       scans the DB to find all modules in the TOB and generates the det_ids.
    **/
    public TOBAnalyzer() throws java.sql.SQLException, java.lang.ClassNotFoundException{
	detIds = new ArrayList<ArrayList<String>>();
	c = CDBConnection.getConnection();
	rodPositionMap = new Hashtable<String, Integer>();

	loadHashTable();
	// OK, now rodPositionMap.get("TOBH.Layer.CS.CR.ROD") gives us the det_id Rod number
	
	if(!DetIDGenerator.mtcc){//static member of DetIDGenerator : is it for the magnet test?
	    ArrayList<ArrayList<String>> v = c.selectQuery("select object_id from cmstrkdb.object_assembly where object='TOB'");
	    if(v.size()==1){
		String tobID = (v.get(0)).get(0);
		getCoolingSegments(tobID);
	    }
	    else{
		if(v.size()==0)
		    throw new java.sql.SQLException("There is no TOB in the database!");
		if(v.size()>1)
		    throw new java.sql.SQLException("There are "+v.size()+" TOBs in the database!");
	    }
	}
    }

    /**
       Fill the HashTable with the correspondances between the position as in the DB and the position in the DetId
       The data are coming from the file RodNumbering.csv in the ressources folder of the JAR file
    **/
    private void loadHashTable(){
	BufferedReader input = null;
	try{
	    InputStream stream = this.getClass().getResourceAsStream(ROD_POSITION_FILE);
	    input = new BufferedReader(new InputStreamReader(stream));
	    String line = null;
	    while((line = input.readLine()) != null){
		String[] val = line.split("\\,");
		rodPositionMap.put(val[0], Integer.parseInt(val[2].replaceAll(" ","")));
		rodPositionMap.put(val[1], Integer.parseInt(val[2].replaceAll(" ","")));		
	    }
	}
	catch(java.io.IOException e){
	    System.out.println("Error : "+e.getMessage());
	}
	finally{
	    try{
		if(input!=null)
		    input.close();
	    }
	    catch(java.io.IOException e){
		System.out.println("Error : "+e.getMessage());	
	    }
	}
    }

    /**
       Convert a database position of a ROD (TOBH.L.CS.CR.ROD) into a det_id position (Layer.backForw.ROD)
       @param The database position of a ROD with the format (TOBH.L.CS.CR.ROD)
       @return The position of the ROD in the det_id format (Layer.backForw.ROD)
    **/
    private String databasePositionToDetId(String position) throws java.sql.SQLException{
	Integer rodPosition = rodPositionMap.get(position);//This is the number of the ROD in the Layer	
	if(rodPosition==null)
	    throw new java.sql.SQLException("Invalid ROD position : "+position);
	String[] dbPos = position.split("\\.");
	int half = Integer.parseInt(dbPos[0]);
	if(half==2)
	    half=0;
	half++; // 1:TOB- 2:TOB+
	return(dbPos[1]+"."+half+"."+rodPosition);//Layer.backForw.Rod
    }


    /**
       Search the det_ids for a TOB
       @param tobId The ID of the TOB
    **/
    private void getCoolingSegments(String tobId) throws java.sql.SQLException{
	ArrayList<ArrayList<String>> v = c.selectQuery("select object_id, type from "+
						 "cmstrkdb.object_assembly where container_id="+tobId+
						 " and object='TOBCS' order by number_in_container");
	for(ArrayList<String> element : v){
	    String tobcsID = element.get(0);
	    String tobcsPosition = element.get(1);// The type of the TOBCS is TOBH.Layer.Position_in_layer
	    getControlRings(tobcsID, tobcsPosition);
	}
    }
    
    /**
       Search the det_ids for a cooling segment
       @param tobcsId The ID of the cooling segment
       @param position The position of the cooling segment in the layer (TOBH.L.CS)
    **/
    private void getControlRings(String tobcsId, String position) throws java.sql.SQLException{
	ArrayList<ArrayList<String>> v = c.selectQuery("select object_id, number_in_container from cmstrkdb.object_assembly "+
						 "where container_id="+tobcsId+" and object='TOBCR' order by number_in_container");
	for(ArrayList<String> element : v){
	    String tobcrID = element.get(0);
	    String tobcrPosition = element.get(1);
	    getRods(tobcrID, position+"."+tobcrPosition);
	}
    }

    /**
       Search the det_ids for a control ring
       @param tobcrId The ID of the control ring
       @param position The position of the control ring in the cooling segment (TOBH.L.CS.CR)
    **/
    private void getRods(String tobcrId, String position) throws java.sql.SQLException{
	ArrayList<ArrayList<String>> v = c.selectQuery("select object_id, number_in_container from cmstrkdb.object_assembly "+
						 "where container_id="+tobcrId+" and object='ROD' order by number_in_container");
	for(ArrayList<String> element : v){
	    String rodID = element.get(0);
	    String rodPosition = element.get(1);
	    rodPosition = databasePositionToDetId(position+"."+rodPosition);//Convert to det_id position
	    getMods(rodID, rodPosition);
	}
    }

    /**
       Search the det_ids for a Rod
       @param rodId The ID of the rod
       @param position The position of the rod in the TOB (Layer.bkdFrwd.Rod)
    **/
    private void getMods(String rodId, String position) throws java.sql.SQLException{
	ArrayList<ArrayList<String>> v = c.selectQuery("select object_id, number_in_container from cmstrkdb.object_assembly "+
						 "where container_id="+rodId+" and object='MOD' order by number_in_container");
	for(ArrayList<String> element : v){
	    String modID = element.get(0);
	    String modPosition = element.get(1);

	    int modPos = Integer.parseInt(modPosition);

	    String[] currentPosition = position.split("\\.");
	    int layer = Integer.parseInt(currentPosition[0]);
	    int forwadBackward = Integer.parseInt(currentPosition[1]);

	    boolean glued=false;
	    boolean stereo=false;

	    if(layer<3){
		glued = true;
		
		if(((modPos+(forwadBackward-1)+((modPos-1)/6))%2)==1)
		    stereo=true;
	    }
	    modPos = modPos%6;//Position 7 in DB is position 1
	    if(modPos==0)
		modPos=6;
	    int stereoFlag = 0; //0 mono, 1 stereo, 2 glued
	    if(stereo)
		stereoFlag = 1;
	    else{
		if(glued)
		    stereoFlag = 2;
	    }
	    
	    String detId = TOB_PREFIXE+"."+position+"."+modPos+"."+stereoFlag;
	    ArrayList<String> couple = new ArrayList<String>();
	    couple.add(modID);
	    couple.add(detId);
	    detIds.add(couple);
	}
    }

    public ArrayList<ArrayList<String>> getDetIds(){
	return detIds;
    }

}
