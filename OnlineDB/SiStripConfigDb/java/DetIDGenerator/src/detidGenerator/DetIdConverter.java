package detidGenerator;

/**
 * <p>Used to convert the det id to a 32 bits word</p>
 * @author G. Baulieu
 * @version 1.0
**/

/*
  $Date: 2006/06/28 11:42:24 $
  
  $Log: DetIdConverter.java,v $
  Revision 1.1  2006/06/28 11:42:24  gbaulieu
  First import of the sources

  Revision 1.2  2006/02/10 09:22:13  baulieu
  Retrieve the fibers length

  Revision 1.1  2006/02/08 14:39:59  baulieu
  Converts the Det_id into 32 bits numbers


*/

public class DetIdConverter{

    protected int id;
    private int detector;
    private int subDetector;

    private final short detectorStartBit = 28;
    private final short subDetectorStartBit = 25;

    private short detectorMask = 0xF;
    private short subDetectorMask = 0x7;

    /**
       @param d The detector
       @param sd The subdetector
    **/
    public DetIdConverter(int d, int sd){
	detector = d;
	subDetector = sd;
    }

    /**
       @param detId The detID in 32 bits
    **/
    public DetIdConverter(int detId){
	id = detId;
	detector = getDetector();
	subDetector = getSubDetector();
    }

    /**
       Create the Det ID
       @return A 32 bits integer containing the Det_id
    **/
    public int compact(){
	id = (detector&detectorMask)<<detectorStartBit |
	    (subDetector&subDetectorMask)<<subDetectorStartBit;
	return id;
    }

    /**
       Get the detector number
       @return The detector
    **/
    public int getDetector(){
	return (id>>detectorStartBit)&detectorMask;
    }

    /**
       Get the subDetector number
       @return The subDetector number
    **/
    public int getSubDetector(){
	return (id>>subDetectorStartBit)&subDetectorMask;
    }
/*
    public static void main(String[] args){
	DetIdConverter d = new DetIdConverter(487096710);
	System.out.println(d.getDetector()+"."+d.getSubDetector());
    }
*/
}