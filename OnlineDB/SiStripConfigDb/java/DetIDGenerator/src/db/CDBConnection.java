package db;

import java.sql.*;
import java.util.Vector;

/**
 * <p>Used to create a connection to a database and perform queries</p>
 * @author G. Baulieu
 * @version 1.0
**/
/*
  $Date: 2006/06/28 11:42:24 $
  
  $Log: CDBConnection.java,v $
  Revision 1.1  2006/06/28 11:42:24  gbaulieu
  First import of the sources

  Revision 1.4  2006/02/02 17:17:00  baulieu
  Some modifications for JDK 1.5
  Call a PL/SQL function to export the parameters

  Revision 1.3  2006/02/01 18:31:50  baulieu
  Allow to execute PL/SQL functions
  
  Revision 1.2  2006/01/30 10:35:00  baulieu
  Ajoute un entete a tous les fichiers
*/

public class CDBConnection
{

    /**Handler on the class **/
    private static CDBConnection handler=null;

    private Connection connect;
    private String currentQuery;
    private Vector<Savepoint> savePointList;
    private String user;
    private String password;
    private String url;
    private boolean connected;


    /**
       Give a handler on the class. You can get it from anywhere with CDBConnection.getConnection()
       @return The current CDBConnection object or null if there is no current connection
    **/
    public static CDBConnection getConnection() throws java.lang.ClassNotFoundException{
	if(handler!=null)
	    return handler;
	else{
	    CDBConnection c = new CDBConnection();
	    handler = c;
	    return c;
	}
    }

    /**
     * Default constructor
     * <p>Throws ClassNotFoundException if Oracle Driver can not be found
     */
    private CDBConnection() throws java.lang.ClassNotFoundException
    {
	try{
	    Class.forName("oracle.jdbc.driver.OracleDriver");
	    user="";
	    password="";
	    url = "";
	    connected = false;
	    currentQuery = "";
	    savePointList = new Vector<Savepoint>();
	}
	catch(java.lang.ClassNotFoundException e){
	    throw new ClassNotFoundException(e.getMessage());
	}
    }

    /**
     * Constructor
     * <p>Throws ClassNotFoundException if Oracle Driver can not be found
     * @param pUser user name
     * @param pPassword database password
     * @param pUrl URL of the database
     */
    private CDBConnection(String pUser, String pPassword, String pUrl) throws java.lang.ClassNotFoundException
    {
	try{
	    Class.forName("oracle.jdbc.driver.OracleDriver");
	    user=pUser;
	    password=pPassword;
	    url = pUrl;
	    connected = false;
	    currentQuery = "";
	    savePointList = new Vector<Savepoint>();
	}
	catch(java.lang.ClassNotFoundException e){
	    throw new ClassNotFoundException(e.getMessage());
	}
    }

    /**
     * Constructor
     * @param c A Connection object
     */
    public CDBConnection(Connection c) throws java.sql.SQLException
    {
	if(c!=null){
	    user=c.getMetaData().getUserName();
	    password="";
	    url = c.getMetaData().getURL();
	    connected = !c.isClosed();
	    connect = c;
	    handler = this;
	    currentQuery = "";
	    savePointList = new Vector<Savepoint>();
	}
	else
	    throw new java.sql.SQLException("Not a valid java.sql.Connection object");
    }

    /**
     * 
     * Connect to the db
     * <p>Throws SQLException if failed to connect
     *
     */
    public void connect() throws java.sql.SQLException
    {
	try{
	    connect = DriverManager.getConnection(url,user,password);   
	    connected = true;
	    handler = this;
	}
	catch(java.sql.SQLException e){
	    throw new SQLException(e.getMessage());
	}	
    }

     /**
     * 
     * Disconnect from the db (perform a rollback before closing the connection, 
     * all transaction not commited will be lost.)
     * <p>Throws SQLException if failed to discconnect
     *
     */
    public void disconnect() throws java.sql.SQLException
    {
	if(isConnected()){
	    try{
		rollback();
		connect.close();   
		connected = false;
	    }
	    catch(java.sql.SQLException e){
		throw new SQLException(e.getMessage());
	    }	
	}
    }
    
    /**
     * Change the user
     * @param pUser new user name
     */
    public void setUser(String pUser)
    {
	user = pUser;
    }

    /**
     * Change the password
     * @param pPassword new password
     */
    public void setPassword(String pPassword)
    {
	password = pPassword;
    }

    /**
     * Change the url
     * @param pUrl new url
     */
    public void setUrl(String pUrl)
    {
	url = pUrl;
    }

    /**
     * Check if the connection is active
     * @return true if connected
     */
    public boolean isConnected()
    {
	return connected;
    }


    /**
     * Begin a new transaction
     **/
    public void beginTransaction() throws SQLException
    {
	try{
	    connect.setAutoCommit(false);
	    savePointList.clear();
	}
	catch(java.sql.SQLException e){
	    throw new SQLException(e.getMessage());
	}
    }

    /**
     * Set a savepoint
     **/
    public void setSavePoint() throws SQLException 
    {	
	try{
	    Savepoint savepoint = connect.setSavepoint();
	    savePointList.add(savepoint);
	}
	catch(java.sql.SQLException e){
	    throw new SQLException(e.getMessage());
	}
    }

    /**
       Delete the last savepoint
    **/
    public void deleteLastSavePoint()
    {	
	if(savePointList.size()>0){
	    savePointList.removeElementAt(savePointList.size()-1);
	}
    }

    /**
       Cancel all modifications done since the given savepoint. The savepoint and all following are deleted.
       @param level The number of the savepoint (starting from the last one)
    **/
    public void rollbackToSavePoint(int level) throws SQLException 
    {	
	try{
	    if(level>0 && level<=savePointList.size()){
		Savepoint sp;
		sp = (Savepoint)savePointList.get(savePointList.size()-level);
		connect.rollback(sp);
		for(int i=0;i<level;i++){
		    savePointList.removeElementAt(savePointList.size()-1);
		}
	    }
	    else{
		throw new SQLException("There is no such savePoint : level "+level+" on "
				       +savePointList.size()+" savepoints");
	    }
	}
	catch(java.sql.SQLException e){
	    throw new SQLException(e.getMessage());
	}
    }

    /**
       Cancel all modifications since the begining of the transaction
    **/
    public void rollback() throws SQLException
    {
	try{
	    connect.rollback();
	    savePointList.clear();
	}
	catch(java.sql.SQLException e){
	    throw new SQLException(e.getMessage());
	}
    }

    /**
     * Commit all the modifications
     */
    public void commit() throws SQLException
    {
	try{
	    connect.commit();
	    savePointList.clear();
	}
	catch(java.sql.SQLException e){
	    throw new SQLException(e.getMessage());
	}
    }

    /**
     * Give the last executed query
     * @return A String containing the last executed query
     */
    public String getLastQuery(){
	return currentQuery;
    }

    /**
     * Execute a SELECT query
     * <p>Throws SQLException if the query failed
     * @param query The query to execute
     * @return a vector of vectors. Each vector is a row
     */
    public synchronized Vector<Vector<String>> selectQuery(String query) throws SQLException
    {
	if(isConnected()){
	    Vector<Vector<String>> result = new Vector<Vector<String>>();
	    int i = 0;
	    
	    try{
		currentQuery = query;
		Statement stmt = connect.createStatement();
		ResultSet rs = stmt.executeQuery(query);
		ResultSetMetaData md;
		
		while (rs.next()) {
		    Vector<String> v = new Vector<String>();
		    md = rs.getMetaData();	
		    for(i=1;i<=md.getColumnCount();i++)
		    {    
			String s = rs.getString(md.getColumnName(i));
			v.add(s);
		    }
		    result.add(v);
		}
		stmt.close();
		return result;
	    }
	    catch(SQLException e)
	    {
		throw new SQLException(e.getMessage());
	    }  
	}
	else{
	    throw new SQLException("Not Connected");
	}
    }

    /**
     * Execute a INSERT / UPDATE / DELETE query
     * <p>Throws SQLException if the query failed
     * @param query The query to execute
     */
    public synchronized void executeQuery(String query) throws SQLException
    {
	if(isConnected()){
	    try{
		currentQuery = query;
		Statement stmt = connect.createStatement();
		stmt.executeUpdate(query);
		stmt.close();
	    }
	    catch(SQLException e)
	    {
		throw new SQLException(e.getMessage());
	    }  
	}
	else{
	    throw new SQLException("Not Connected");
	}
    }

/**
     * Call a PL/SQL function
     * <p>Throws SQLException if there is a problem.
     * <p>Throws ClassNotSupportedException if a parameter is not supported
     * @param functionName The name of the function
     * @param parameters A list of parameters (can be int, String, float, double cast into Object or null)
     * @return The result of the function (must be an int)
     */
    public synchronized int callFunction(String functionName, Object... parameters) 
	throws SQLException, ClassNotSupportedException{
	
	int result = -1;
	String proc_schem="begin ? :="+functionName+" (";
	for (int i=0;i<parameters.length-1;i++){
	    proc_schem+="?,";
	}
	proc_schem+="?); end;";
	
	CallableStatement cs = connect.prepareCall(proc_schem);
	cs.registerOutParameter(1,Types.INTEGER);
	//set the parameters
	for(int i=0; i<parameters.length; i++){
	    if (parameters[i]==null){
		cs.setString(i+2,"");
	    }
	    else{
		if(parameters[i].getClass().getName().equals("java.lang.String")){
		    cs.setString(i+2,(String)parameters[i]);
		}
		else{
		    if(parameters[i].getClass().getName().equals("java.lang.Integer")){
			cs.setInt(i+2,(Integer)parameters[i]);
		    }
		    else{
			if(parameters[i].getClass().getName().equals("java.lang.Double")){
			    cs.setDouble(i+2,(Double)parameters[i]);
			}
			else{
			    if(parameters[i].getClass().getName().equals("java.lang.Float")){
				cs.setFloat(i+2,(Float)parameters[i]);
			    }
			    else{
				throw new ClassNotSupportedException("Parameter : "+parameters[i]+
								     " of type "+parameters[i].getClass().getName()+
								     " is not supported");
			    }
			}
		    }
		}
	    }
	}
	cs.execute();
	result=cs.getInt(1);
	cs.close();
	return result;
    }
}
