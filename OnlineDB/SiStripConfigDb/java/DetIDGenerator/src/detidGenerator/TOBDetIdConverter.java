package detidGenerator;

/**
 * <p>Used to convert the det id to a 32 bits word</p>
 * @author G. Baulieu
 * @version 1.0
**/

/*
  $Date: 2006/02/08 15:03:00 $
  
  $Log: TOBDetIdConverter.java,v $
  Revision 1.1  2006/02/08 15:03:00  baulieu
  Add the convertion to 32 bits for the TOB


*/

public class TOBDetIdConverter extends DetIdConverter{

    private int layer;
    private int frontBack;
    private int rod;
    private int moduleNumber;
    private int stereo;

    private final short layerStartBit =     16;
    private final short rod_fw_bwStartBit = 15;
    private final short rodStartBit =        8;
    private final short detStartBit =        2;
    private final short sterStartBit =       0;

    private final short layerMask =       0xF;
    private final short rod_fw_bwMask =   0x1;
    private final short rodMask =         0x7F;
    private final short detMask =         0x3F;
    private final short sterMask =        0x3;

    public TOBDetIdConverter(int l, int fb, int r, int mn, int s){
	super(1, 5);
	layer = l;
	frontBack = fb;
	rod = r;
	moduleNumber = mn;
	stereo = s;
    }

    public TOBDetIdConverter(String detID) throws Exception{
	super(1,5);
	try{
	    String[] val = detID.split("\\.");
	    if(val.length!=8)
		throw new Exception("The detID has an invalid format");
	    else{
		layer = Integer.parseInt(val[3]);
		frontBack = Integer.parseInt(val[4]);
		rod = Integer.parseInt(val[5]);
		moduleNumber = Integer.parseInt(val[6]);
		stereo = Integer.parseInt(val[7]);
	    }
	}
	catch(NumberFormatException e){
	    throw new Exception("TECDetIdConverter : \n"+e.getMessage());
	}
    }

    public int compact(){
	super.compact();
	id |= (layer&layerMask)<<layerStartBit |
	    (frontBack&rod_fw_bwMask)<<rod_fw_bwStartBit |
	    (rod&rodMask)<<rodStartBit |
	    (moduleNumber&detMask)<<detStartBit |
	    (stereo&sterMask)<<sterStartBit;
	return id;
    }

    public int getLayer(){
	return (id>>layerStartBit)&layerMask;
    }

    public int getFrontBack(){
	return (id>>rod_fw_bwStartBit)&rod_fw_bwMask;
    }

    public int getRod(){
	return (id>>rodStartBit)&rodMask;
    }

    public int getModNumber(){
	return (id>>detStartBit)&detMask;
    }

    public int getStereo(){
	return (id>>sterStartBit)&sterMask;
    }

}