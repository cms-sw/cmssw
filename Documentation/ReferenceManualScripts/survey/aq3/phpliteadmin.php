<?php

//
//  Project: phpLiteAdmin (http://phpliteadmin.googlecode.com)
//  Version: 1.9.2
//  Summary: PHP-based admin tool to manage SQLite2 and SQLite3 databases on the web
//  Last updated: 5/30/12
//  Developers:
//     Dane Iracleous (daneiracleous@gmail.com)
//     Ian Aldrighetti (ian.aldrighetti@gmail.com)
//     George Flanagin & Digital Gaslight, Inc (george@digitalgaslight.com)
//		 Christopher Kramer (crazy4chrissi@gmail.com)
//
//
//  Copyright (C) 2011  phpLiteAdmin
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
///////////////////////////////////////////////////////////////////////////

//please report any bugs you encounter to http://code.google.com/p/phpliteadmin/issues/list


//BEGIN USER-DEFINED VARIABLES
//////////////////////////////

//password to gain access
$password = "n3v3rkn0wwhy";

//directory relative to this file to search for databases (if false, manually list databases in the $databases variable)
$directory = ".";

//whether or not to scan the subdirectories of the above directory infinitely deep
$subdirectories = false;

//if the above $directory variable is set to false, you must specify the databases manually in an array as the next variable
//if any of the databases do not exist as they are referenced by their path, they will be created automatically
$databases = array
(
	array
	(
		"path"=> "aq.db",
		"name"=> "Database 1"
	)
//,
//	array
//	(
//		"path"=> "database2.sqlite",
//		"name"=> "Database 2"
//	)
);

//a list of custom functions that can be applied to columns in the databases
//make sure to define every function below if it is not a core PHP function
$custom_functions = array('md5', 'md5rev', 'sha1', 'sha1rev', 'time', 'mydate', 'strtotime', 'myreplace');

//define all the non-core custom functions
function md5rev($value)
{
	return strrev(md5($value));
}
function sha1rev($value)
{
	return strrev(sha1($value));
}
function mydate($value)
{
	return date("g:ia n/j/y", intval($value));
}
function myreplace($value)
{
	return ereg_replace("[^A-Za-z0-9]", "", strval($value));	
}

//changing the following variable allows multiple phpLiteAdmin installs to work under the same domain.
$cookie_name = 'pla3412';

//whether or not to put the app in debug mode where errors are outputted
$debug = false;

////////////////////////////
//END USER-DEFINED VARIABLES


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//there is no reason for the average user to edit anything below this comment
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

session_start(); //don't mess with this - required for the login session
date_default_timezone_set(date_default_timezone_get()); //needed to fix STRICT warnings about timezone issues

if($debug==true)
{
	ini_set("display_errors", 1);
	error_reporting(E_STRICT | E_ALL);
}

$startTimeTot = microtime(true); //start the timer to record page load time

//the salt and password encrypting is probably unnecessary protection but is done just for the sake of being very secure
//create a random salt for this session if a cookie doesn't already exist for it
if(!isset($_SESSION[$cookie_name.'_salt']) && !isset($_COOKIE[$cookie_name.'_salt']))
{
	$n = rand(10e16, 10e20);
	$_SESSION[$cookie_name.'_salt'] = base_convert($n, 10, 36);
}
else if(!isset($_SESSION[$cookie_name.'_salt']) && isset($_COOKIE[$cookie_name.'_salt'])) //session doesn't exist, but cookie does so grab it
{
	$_SESSION[$cookie_name.'_salt'] = $_COOKIE[$cookie_name.'_salt'];
}

//build the basename of this file for later reference
$info = pathinfo($_SERVER['PHP_SELF']);
$thisName = $info['basename'];

//constants
define("PROJECT", "phpLiteAdmin");
define("VERSION", "1.9.2");
define("PAGE", $thisName);
define("COOKIENAME", $cookie_name);
define("SYSTEMPASSWORD", $password); // Makes things easier.
define("SYSTEMPASSWORDENCRYPTED", md5($password."_".$_SESSION[$cookie_name.'_salt'])); //extra security - salted and encrypted password used for checking
define("FORCETYPE", false); //force the extension that will be used (set to false in almost all circumstances except debugging)

//data types array
$types = array("INTEGER", "REAL", "TEXT", "BLOB");
define("DATATYPES", serialize($types));

//accepted db extensions
$exts = array("sqlite", "sqlite3", "db", "db3");
define("EXTENSIONS", serialize($exts));

//available SQLite functions array (don't add anything here or there will be problems)
$functions = array("abs", "hex", "length", "lower", "ltrim", "random", "round", "rtrim", "trim", "typeof", "upper");
define("FUNCTIONS", serialize($functions));
define("CUSTOM_FUNCTIONS", serialize($custom_functions));

//function that allows SQL delimiter to be ignored inside comments or strings
function explode_sql($delimiter, $sql)
{
	$ign = array('"' => '"', "'" => "'", "/*" => "*/", "--" => "\n"); // Ignore sequences.
	$out = array();
	$last = 0;
	$slen = strlen($sql);
	$dlen = strlen($delimiter);
	$i = 0;
	while($i < $slen)
	{
		// Split on delimiter
		if($slen - $i >= $dlen && substr($sql, $i, $dlen) == $delimiter)
		{
			array_push($out, substr($sql, $last, $i - $last));
			$last = $i + $dlen;
			$i += $dlen;
			continue;
		}
		// Eat comments and string literals
		foreach($ign as $start => $end)
		{
			$ilen = strlen($start);
			if($slen - $i >= $ilen && substr($sql, $i, $ilen) == $start)
			{
				$i+=strlen($start);
				$elen = strlen($end);
				while($i < $slen)
				{
					if($slen - $i >= $elen && substr($sql, $i, $elen) == $end)
					{
						// SQL comment characters can be escaped by doubling the character. This recognizes and skips those.
						if($start == $end && $slen - $i >= $elen*2 && substr($sql, $i, $elen*2) == $end.$end)
						{
							$i += $elen * 2;
							continue;
						}
						else
						{
							$i += $elen;
							continue 3;
						}
					}
					$i++;
				}
				continue 2;
			}		
		}
		$i++;
	}
	if($last < $slen)
		array_push($out, substr($sql, $last, $slen - $last));
	return $out;
}

//function to scan entire directory tree and subdirectories
function dir_tree($dir)
{
	$path = '';
	$stack[] = $dir;
	while($stack)
	{
		$thisdir = array_pop($stack);
		if($dircont = scandir($thisdir))
		{
			$i=0;
			while(isset($dircont[$i]))
			{
				if($dircont[$i] !== '.' && $dircont[$i] !== '..')
				{
					$current_file = "{$thisdir}/{$dircont[$i]}";
					if(is_file($current_file))
					{
						$path[] = "{$thisdir}/{$dircont[$i]}";
					}
					elseif (is_dir($current_file))
					{
						$path[] = "{$thisdir}/{$dircont[$i]}";
						$stack[] = $current_file;
					}
				}
				$i++;
			}
		}
	}
	return $path;
}

//the function echo the help [?] links to the documentation
function helpLink($name)
{
	return "<a href='javascript:openHelp(\"".$name."\");' class='helpq' title='Help: ".$name."'>[?]</a>";	
}

//user is deleting a database
if(isset($_GET['database_delete']))
{
	$dbpath = $_POST['database_delete'];
	unlink($dbpath);
	$_SESSION[COOKIENAME.'currentDB'] = 0;
}

//user is renaming a database
if(isset($_GET['database_rename']))
{
	$oldpath = $_POST['oldname'];
	$newpath = $_POST['newname'];
	if(!file_exists($newpath))
	{
		copy($oldpath, $newpath);
		unlink($oldpath);
		$justrenamed = true;
	}
	else
	{
		$dbexists = true;	
	}
}

//user is creating a new Database
if(isset($_POST['new_dbname']))
{
	$str = preg_replace('@[^\w-.]@','', $_POST['new_dbname']);
	$dbname = $str;
	$dbpath = $str;
	$info = pathinfo($dbpath);
	if(!isset($info['extension']))
		$dbpath = $dbpath.".".$exts[0];
	else
	{
		if(!in_array(strtolower($info['extension']), $exts))
		{
			$dbpath = $dbpath.".".$exts[0];
		}
	}
	$tdata = array();	
	$tdata['name'] = $dbname;
	$tdata['path'] = $directory."/".$dbpath;
	$td = new Database($tdata);
}

//if the user wants to scan a directory for databases, do so
if($directory!==false)
{
	if($directory[strlen($directory)-1]=="/") //if user has a trailing slash in the directory, remove it
		$directory = substr($directory, 0, strlen($directory)-1);
		
	if(is_dir($directory)) //make sure the directory is valid
	{
		if($subdirectories===true)
			$arr = dir_tree($directory);
		else
			$arr = scandir($directory);
		$databases = array();
		$j = 0;
		for($i=0; $i<sizeof($arr); $i++) //iterate through all the files in the databases
		{
			$file = pathinfo($arr[$i]);
			if(isset($file['extension']))
			{
				$ext = strtolower($file['extension']);
				if(in_array(strtolower($ext), $exts)) //make sure the file is a valid SQLite database by checking its extension
				{
					if($subdirectories===true)
						$databases[$j]['path'] = $arr[$i];
					else
						$databases[$j]['path'] = $directory."/".$arr[$i];
					$databases[$j]['name'] = $arr[$i];
					// 22 August 2011: gkf fixed bug 49.
					$perms = 0;
					$perms += is_readable($databases[$j]['path']) ? 4 : 0;
					$perms += is_writeable($databases[$j]['path']) ? 2 : 0;
					switch($perms)
					{
						case 6: $perms = "[rw] "; break;
						case 4: $perms = "[r ] "; break;
						case 2: $perms = "[ w] "; break; // God forbid, but it might happen.
						default: $perms = "[  ] "; break;
					}
					$databases[$j]['perms'] = $perms;
					$j++;
				}
			}
		}
		// 22 August 2011: gkf fixed bug #50.
		sort($databases);
		if(isset($tdata))
		{
			for($i=0; $i<sizeof($databases); $i++)
			{
				if($tdata['path'] == $databases[$i]['path'])
				{
					$_SESSION[COOKIENAME.'currentDB'] = $i;
					break;
				}
			}
		}
		
		if(isset($justrenamed))
		{
			for($i=0; $i<sizeof($databases); $i++)
			{
				if($newpath == $databases[$i]['path'])
				{
					$_SESSION[COOKIENAME.'currentDB'] = $i;
					break;
				}
			}	
		}
	}
	else //the directory is not valid - display error and exit
	{
		echo "<div class='confirm' style='margin:20px;'>";
		echo "The directory you specified to scan for databases does not exist or is not a directory.";
		echo "</div>";
		exit();
	}
}

// 22 August 2011: gkf added this function to support display of
//                 default values in the form used to INSERT new data.
function deQuoteSQL($s)
{
	return trim(trim($s), "'");
}

//
// Authorization class
// Maintains user's logged-in state and security of application
//
class Authorization
{
	public function grant($remember)
	{
		if($remember) //user wants to be remembered, so set a cookie
		{
			$expire = time()+60*60*24*30; //set expiration to 1 month from now
			setcookie(COOKIENAME, SYSTEMPASSWORD, $expire);
			setcookie(COOKIENAME."_salt", $_SESSION[COOKIENAME.'_salt'], $expire);
		}
		else
		{
			//user does not want to be remembered, so destroy any potential cookies
			setcookie(COOKIENAME, "", time()-86400);
			setcookie(COOKIENAME."_salt", "", time()-86400);
			unset($_COOKIE[COOKIENAME]);
			unset($_COOKIE[COOKIENAME.'_salt']);
		}

		$_SESSION[COOKIENAME.'password'] = SYSTEMPASSWORDENCRYPTED;
	}
	public function revoke()
	{
		//destroy everything - cookies and session vars
		setcookie(COOKIENAME, "", time()-86400);
		setcookie(COOKIENAME."_salt", "", time()-86400);
		unset($_COOKIE[COOKIENAME]);
		unset($_COOKIE[COOKIENAME.'_salt']);
		session_unset();
		session_destroy();
	}
	public function isAuthorized()
	{
		// Is this just session long? (What!?? -DI)
		if((isset($_SESSION[COOKIENAME.'password']) && $_SESSION[COOKIENAME.'password'] == SYSTEMPASSWORDENCRYPTED) || (isset($_COOKIE[COOKIENAME]) && isset($_COOKIE[COOKIENAME.'_salt']) && md5($_COOKIE[COOKIENAME]."_".$_COOKIE[COOKIENAME.'_salt']) == SYSTEMPASSWORDENCRYPTED))
			return true;
		else
		{
			return false;
		}
	}
}

//
// Database class
// Generic database abstraction class to manage interaction with database without worrying about SQLite vs. PHP versions
//
class Database
{
	protected $db; //reference to the DB object
	protected $type; //the extension for PHP that handles SQLite
	protected $data;
	protected $lastResult;
	protected $fns;

	public function __construct($data)
	{
		$this->data = $data;
		$this->fns = array();
		try
		{
			if(!file_exists($this->data["path"]) && !is_writable(dirname($this->data["path"]))) //make sure the containing directory is writable if the database does not exist
			{
				echo "<div class='confirm' style='margin:20px;'>";
				echo "The database, '".$this->data["path"]."', does not exist and cannot be created because the containing directory, '".dirname($this->data["path"])."', is not writable. The application is unusable until you make it writable.";
				echo "<form action='".PAGE."' method='post'/>";
				echo "<input type='submit' value='Log Out' name='logout' class='btn'/>";
				echo "</form>";
				echo "</div><br/>";
				exit();
			}

			$ver = $this->getVersion();

			switch(true)
			{
				case (FORCETYPE=="PDO" || ((FORCETYPE==false || $ver!=-1) && class_exists("PDO") && ($ver==-1 || $ver==3))):
					$this->db = new PDO("sqlite:".$this->data['path']);
					if($this->db!=NULL)
					{
						$this->type = "PDO";
						$cfns = unserialize(CUSTOM_FUNCTIONS);
						for($i=0; $i<sizeof($cfns); $i++)
						{
							$this->db->sqliteCreateFunction($cfns[$i], $cfns[$i], 1);
							$this->addUserFunction($cfns[$i]);	
						}
						break;
					}
				case (FORCETYPE=="SQLite3" || ((FORCETYPE==false || $ver!=-1) && class_exists("SQLite3") && ($ver==-1 || $ver==3))):
					$this->db = new SQLite3($this->data['path']);
					if($this->db!=NULL)
					{
						$cfns = unserialize(CUSTOM_FUNCTIONS);
						for($i=0; $i<sizeof($cfns); $i++)
						{
							$this->db->createFunction($cfns[$i], $cfns[$i], 1);
							$this->addUserFunction($cfns[$i]);	
						}
						$this->type = "SQLite3";
						break;
					}
				case (FORCETYPE=="SQLiteDatabase" || ((FORCETYPE==false || $ver!=-1) && class_exists("SQLiteDatabase") && ($ver==-1 || $ver==2))):
					$this->db = new SQLiteDatabase($this->data['path']);
					if($this->db!=NULL)
					{
						$cfns = unserialize(CUSTOM_FUNCTIONS);
						for($i=0; $i<sizeof($cfns); $i++)
						{
							$this->db->createFunction($cfns[$i], $cfns[$i], 1);
							$this->addUserFunction($cfns[$i]);	
						}
						$this->type = "SQLiteDatabase";
						break;
					}
				default:
					$this->showError();
					exit();
			}
		}
		catch(Exception $e)
		{
			$this->showError();
			exit();
		}
	}
	
	public function getUserFunctions()
	{
		return $this->fns;	
	}
	
	public function addUserFunction($name)
	{
		array_push($this->fns, $name);	
	}
	
	public function getError()
	{
		if($this->type=="PDO")
		{
			$e = $this->db->errorInfo();
			return $e[2];	
		}
		else if($this->type=="SQLite3")
		{
			return $this->db->lastErrorMsg();
		}
		else
		{
			return sqlite_error_string($this->db->lastError());
		}
	}
	
	public function showError()
	{
		$classPDO = class_exists("PDO");
		$classSQLite3 = class_exists("SQLite3");
		$classSQLiteDatabase = class_exists("SQLiteDatabase");
		if($classPDO)
			$strPDO = "installed";
		else
			$strPDO = "not installed";
		if($classSQLite3)
			$strSQLite3 = "installed";
		else
			$strSQLite3 = "not installed";
		if($classSQLiteDatabase)
			$strSQLiteDatabase = "installed";
		else
			$strSQLiteDatabase = "not installed";
		echo "<div class='confirm' style='margin:20px;'>";
		echo "There was a problem setting up your database, ".$this->getPath().". An attempt will be made to find out what's going on so you can fix the problem more easily.<br/><br/>";
		echo "<i>Checking supported SQLite PHP extensions...<br/><br/>";
		echo "<b>PDO</b>: ".$strPDO."<br/>";
		echo "<b>SQLite3</b>: ".$strSQLite3."<br/>";
		echo "<b>SQLiteDatabase</b>: ".$strSQLiteDatabase."<br/><br/>...done.</i><br/><br/>";
		if(!$classPDO && !$classSQLite3 && !$classSQLiteDatabase)
			echo "It appears that none of the supported SQLite library extensions are available in your installation of PHP. You may not use ".PROJECT." until you install at least one of them.";
		else
		{
			if(!$classPDO && !$classSQLite3 && $this->getVersion()==3)
				echo "It appears that your database is of SQLite version 3 but your installation of PHP does not contain the necessary extensions to handle this version. To fix the problem, either delete the database and allow ".PROJECT." to create it automatically or recreate it manually as SQLite version 2.";
			else if(!$classSQLiteDatabase && $this->getVersion()==2)
				echo "It appears that your database is of SQLite version 2 but your installation of PHP does not contain the necessary extensions to handle this version. To fix the problem, either delete the database and allow ".PROJECT." to create it automatically or recreate it manually as SQLite version 3.";
			else
				echo "The problem cannot be diagnosed properly. Please email me at daneiracleous@gmail.com with your database as an attachment and the contents of this error message. It may be that your database is simply not a valid SQLite database, but this is not certain.";
		}
		echo "</div><br/>";
	}

	public function __destruct()
	{
		if($this->db)
			$this->close();
	}

	//get the exact PHP extension being used for SQLite
	public function getType()
	{
		return $this->type;
	}

	//get the name of the database
	public function getName()
	{
		return $this->data["name"];
	}

	//get the filename of the database
	public function getPath()
	{
		return $this->data["path"];
	}

	//get the version of the database
	public function getVersion()
	{
		if(file_exists($this->data['path'])) //make sure file exists before getting its contents
		{
			$content = strtolower(file_get_contents($this->data['path'], NULL, NULL, 0, 40)); //get the first 40 characters of the database file
			$p = strpos($content, "** this file contains an sqlite 2"); //this text is at the beginning of every SQLite2 database
			if($p!==false) //the text is found - this is version 2
				return 2;
			else
				return 3;
		}
		else //return -1 to indicate that it does not exist and needs to be created
		{
			return -1;
		}
	}

	//get the size of the database
	public function getSize()
	{
		return round(filesize($this->data["path"])*0.0009765625, 1)." Kb";
	}

	//get the last modified time of database
	public function getDate()
	{
		return date("g:ia \o\\n F j, Y", filemtime($this->data["path"]));
	}

	//get number of affected rows from last query
	public function getAffectedRows()
	{
		if($this->type=="PDO")
			return $this->lastResult->rowCount();
		else if($this->type=="SQLite3")
			return $this->db->changes();
		else if($this->type=="SQLiteDatabase")
			return $this->db->changes();
	}

	public function close()
	{
		if($this->type=="PDO")
			$this->db = NULL;
		else if($this->type=="SQLite3")
			$this->db->close();
		else if($this->type=="SQLiteDatabase")
			$this->db = NULL;
	}

	public function beginTransaction()
	{
		$this->query("BEGIN");
	}

	public function commitTransaction()
	{
		$this->query("COMMIT");
	}

	public function rollbackTransaction()
	{
		$this->query("ROLLBACK");
	}

	//generic query wrapper
	public function query($query, $ignoreAlterCase=false)
	{
		if(strtolower(substr(ltrim($query),0,5))=='alter' && $ignoreAlterCase==false) //this query is an ALTER query - call the necessary function
		{
			$queryparts = preg_split("/[\s]+/", $query, 4, PREG_SPLIT_NO_EMPTY);
			$tablename = $queryparts[2];
			$alterdefs = $queryparts[3];
			//echo $query;
			$result = $this->alterTable($tablename, $alterdefs);
		}
		else //this query is normal - proceed as normal
			$result = $this->db->query($query);
		if(!$result)
			return NULL;
		$this->lastResult = $result;
		return $result;
	}

	//wrapper for an INSERT and returns the ID of the inserted row
	public function insert($query)
	{
		$result = $this->query($query);
		if($this->type=="PDO")
			return $this->db->lastInsertId();
		else if($this->type=="SQLite3")
			return $this->db->lastInsertRowID();
		else if($this->type=="SQLiteDatabase")
			return $this->db->lastInsertRowid();
	}

	//returns an array for SELECT
	public function select($query, $mode="both")
	{
		$result = $this->query($query);
		if(!$result) //make sure the result is valid
			return NULL;
		if($this->type=="PDO")
		{
			if($mode=="assoc")
				$mode = PDO::FETCH_ASSOC;
			else if($mode=="num")
				$mode = PDO::FETCH_NUM;
			else
				$mode = PDO::FETCH_BOTH;
			return $result->fetch($mode);
		}
		else if($this->type=="SQLite3")
		{
			if($mode=="assoc")
				$mode = SQLITE3_ASSOC;
			else if($mode=="num")
				$mode = SQLITE3_NUM;
			else
				$mode = SQLITE3_BOTH;
			return $result->fetchArray($mode);
		}
		else if($this->type=="SQLiteDatabase")
		{
			if($mode=="assoc")
				$mode = SQLITE_ASSOC;
			else if($mode=="num")
				$mode = SQLITE_NUM;
			else
				$mode = SQLITE_BOTH;
			return $result->fetch($mode);
		}
	}

	//returns an array of arrays after doing a SELECT
	public function selectArray($query, $mode="both")
	{
		$result = $this->query($query);
		if(!$result) //make sure the result is valid
			return NULL;
		if($this->type=="PDO")
		{
			if($mode=="assoc")
				$mode = PDO::FETCH_ASSOC;
			else if($mode=="num")
				$mode = PDO::FETCH_NUM;
			else
				$mode = PDO::FETCH_BOTH;
			return $result->fetchAll($mode);
		}
		else if($this->type=="SQLite3")
		{
			if($mode=="assoc")
				$mode = SQLITE3_ASSOC;
			else if($mode=="num")
				$mode = SQLITE3_NUM;
			else
				$mode = SQLITE3_BOTH;
			$arr = array();
			$i = 0;
			while($res = $result->fetchArray($mode))
			{
				$arr[$i] = $res;
				$i++;
			}
			return $arr;
		}
		else if($this->type=="SQLiteDatabase")
		{
			if($mode=="assoc")
				$mode = SQLITE_ASSOC;
			else if($mode=="num")
				$mode = SQLITE_NUM;
			else
				$mode = SQLITE_BOTH;
			return $result->fetchAll($mode);
		}
	}

	//function that is called for an alter table statement in a query
	//code borrowed with permission from http://code.jenseng.com/db/
	public function alterTable($table, $alterdefs)
	{
		if($alterdefs != '')
		{
			$tempQuery = "SELECT sql,name,type FROM sqlite_master WHERE tbl_name = '".$table."' ORDER BY type DESC";
			$result = $this->query($tempQuery);
			$resultArr = $this->selectArray($tempQuery);

			if(sizeof($resultArr)>0)
			{
				$row = $this->select($tempQuery); //table sql
				$tmpname = 't'.time();
				$origsql = trim(preg_replace("/[\s]+/", " ", str_replace(",", ", ",preg_replace("/[\(]/", "( ", $row['sql'], 1))));
				$createtemptableSQL = 'CREATE TEMPORARY '.substr(trim(preg_replace("'".$table."'", $tmpname, $origsql, 1)), 6);
				$createindexsql = array();
				$i = 0;
				$defs = preg_split("/[,]+/",$alterdefs, -1, PREG_SPLIT_NO_EMPTY);
				$prevword = $table;
				$oldcols = preg_split("/[,]+/", substr(trim($createtemptableSQL), strpos(trim($createtemptableSQL), '(')+1), -1, PREG_SPLIT_NO_EMPTY);
				$newcols = array();
				for($i=0; $i<sizeof($oldcols); $i++)
				{
					$colparts = preg_split("/[\s]+/", $oldcols[$i], -1, PREG_SPLIT_NO_EMPTY);
					$oldcols[$i] = $colparts[0];
					$newcols[$colparts[0]] = $colparts[0];
				}
				$newcolumns = '';
				$oldcolumns = '';
				reset($newcols);
				while(list($key, $val) = each($newcols))
				{
					$newcolumns .= ($newcolumns?', ':'').$val;
					$oldcolumns .= ($oldcolumns?', ':'').$key;
				}
				$copytotempsql = 'INSERT INTO '.$tmpname.'('.$newcolumns.') SELECT '.$oldcolumns.' FROM '.$table;
				$dropoldsql = 'DROP TABLE '.$table;
				$createtesttableSQL = $createtemptableSQL;
				foreach($defs as $def)
				{
					$defparts = preg_split("/[\s]+/", $def,-1, PREG_SPLIT_NO_EMPTY);
					$action = strtolower($defparts[0]);
					switch($action)
					{
						case 'add':
							if(sizeof($defparts) <= 2)
								return false;
							$createtesttableSQL = substr($createtesttableSQL, 0, strlen($createtesttableSQL)-1).',';
							for($i=1;$i<sizeof($defparts);$i++)
								$createtesttableSQL.=' '.$defparts[$i];
							$createtesttableSQL.=')';
							break;
						case 'change':
							if(sizeof($defparts) <= 3)
							{
								return false;
							}
							if($severpos = strpos($createtesttableSQL,' '.$defparts[1].' '))
							{
								if($newcols[$defparts[1]] != $defparts[1])
									return false;
								$newcols[$defparts[1]] = $defparts[2];
								$nextcommapos = strpos($createtesttableSQL,',',$severpos);
								$insertval = '';
								for($i=2;$i<sizeof($defparts);$i++)
									$insertval.=' '.$defparts[$i];
								if($nextcommapos)
									$createtesttableSQL = substr($createtesttableSQL,0,$severpos).$insertval.substr($createtesttableSQL,$nextcommapos);
								else
									$createtesttableSQL = substr($createtesttableSQL,0,$severpos-(strpos($createtesttableSQL,',')?0:1)).$insertval.')';
							}
							else
								return false;
							break;
						case 'drop':
							if(sizeof($defparts) < 2)
								return false;
							if($severpos = strpos($createtesttableSQL,' '.$defparts[1].' '))
							{
								$nextcommapos = strpos($createtesttableSQL,',',$severpos);
								if($nextcommapos)
									$createtesttableSQL = substr($createtesttableSQL,0,$severpos).substr($createtesttableSQL,$nextcommapos + 1);
								else
									$createtesttableSQL = substr($createtesttableSQL,0,$severpos-(strpos($createtesttableSQL,',')?0:1) - 1).')';
								unset($newcols[$defparts[1]]);
							}
							else
								return false;
							break;
						default:
							return false;
					}
					$prevword = $defparts[sizeof($defparts)-1];
				}
				//this block of code generates a test table simply to verify that the columns specifed are valid in an sql statement
				//this ensures that no reserved words are used as columns, for example
				$tempResult = $this->query($createtesttableSQL);
				if(!$tempResult)
					return false;
				$droptempsql = 'DROP TABLE '.$tmpname;
				$tempResult = $this->query($droptempsql);
				//end block

				$createnewtableSQL = 'CREATE '.substr(trim(preg_replace("'".$tmpname."'", $table, $createtesttableSQL, 1)), 17);
				$newcolumns = '';
				$oldcolumns = '';
				reset($newcols);
				while(list($key,$val) = each($newcols))
				{
					$newcolumns .= ($newcolumns?', ':'').$val;
					$oldcolumns .= ($oldcolumns?', ':'').$key;
				}
				$copytonewsql = 'INSERT INTO '.$table.'('.$newcolumns.') SELECT '.$oldcolumns.' FROM '.$tmpname;

				$this->query($createtemptableSQL); //create temp table
				$this->query($copytotempsql); //copy to table
				$this->query($dropoldsql); //drop old table

				$this->query($createnewtableSQL); //recreate original table
				$this->query($copytonewsql); //copy back to original table
				$this->query($droptempsql); //drop temp table
			}
			else
			{
				return false;
			}
			return true;
		}
	}

	//multiple query execution
	public function multiQuery($query)
	{
		$error = "Unknown error.";
		if($this->type=="PDO")
		{
			$success = $this->db->exec($query);
		}
		else if($this->type=="SQLite3")
		{
			$success = $this->db->exec($query, $error);
		}
		else
		{
			$success = $this->db->queryExec($query, $error);
		}
		if(!$success)
		{
			return "Error in query: '".$error."'";
		}
		else
		{
			return true;	
		}
	}

	//get number of rows in table
	public function numRows($table)
	{
		$result = $this->select("SELECT Count(*) FROM ".$table);
		return $result[0];
	}

	//correctly escape a string to be injected into an SQL query
	public function quote($value)
	{
		if($this->type=="PDO")
		{
			// PDO quote() escapes and adds quotes
			return $this->db->quote($value);
		}
		else if($this->type=="SQLite3")
		{
			return "'".$this->db->escapeString($value)."'";
		}
		else
		{
			return "'".sqlite_escape_string($value)."'";
		}
	}

	//correctly format a string value from a table before showing it
	public function formatString($value)
	{
		return htmlspecialchars(stripslashes($value));
	}

	//import sql
	public function import_sql($query)
	{
		return $this->multiQuery($query);
	}
	
	//import csv
	public function import_csv($filename, $table, $field_terminate, $field_enclosed, $field_escaped, $null, $fields_in_first_row)
	{
		// CSV import implemented by Christopher Kramer - http://www.christosoft.de
		$csv_handle = fopen($filename,'r');
		$csv_insert = "BEGIN;\n";
		$csv_number_of_rows = 0;
		// PHP requires enclosure defined, but has no problem if it was not used
		if($field_enclosed=="") $field_enclosed='"';
		// PHP requires escaper defined
		if($field_escaped=="") $field_escaped='\\';
		while(!feof($csv_handle))
		{
			$csv_data = fgetcsv($csv_handle, 0, $field_terminate, $field_enclosed, $field_escaped); 
			if($csv_data[0] != NULL || count($csv_data)>1)
			{
				$csv_number_of_rows++;
				if($fields_in_first_row && $csv_number_of_rows==1) continue; 
				$csv_col_number = count($csv_data);
				$csv_insert .= "INSERT INTO $table VALUES (";
				foreach($csv_data as $csv_col => $csv_cell)
				{
					if($csv_cell == $null) $csv_insert .= "NULL";
					else
					{
						$csv_insert.= $this->quote($csv_cell);
					}
					if($csv_col == $csv_col_number-2 && $csv_data[$csv_col+1]=='')
					{
						// the CSV row ends with the separator (like old phpliteadmin exported)
						break;
					} 
					if($csv_col < $csv_col_number-1) $csv_insert .= ",";
				}
				$csv_insert .= ");\n";
				
				if($csv_number_of_rows > 5000)
				{
					$csv_insert .= "COMMIT;\nBEGIN;\n";
					$csv_number_of_rows = 0;
				}
			}
		}
		$csv_insert .= "COMMIT;";
		fclose($csv_handle);
		return $this->multiQuery($csv_insert);

	}
	
	//export csv
	public function export_csv($tables, $field_terminate, $field_enclosed, $field_escaped, $null, $crlf, $fields_in_first_row)
	{
		$field_enclosed = stripslashes($field_enclosed);
		$query = "SELECT * FROM sqlite_master WHERE type='table' ORDER BY type DESC";
		$result = $this->selectArray($query);
		for($i=0; $i<sizeof($result); $i++)
		{
			$valid = false;
			for($j=0; $j<sizeof($tables); $j++)
			{
				if($result[$i]['tbl_name']==$tables[$j])
					$valid = true;
			}
			if($valid)
			{
				$query = "PRAGMA table_info('".$result[$i]['tbl_name']."')";
				$temp = $this->selectArray($query);
				$cols = array();
				for($z=0; $z<sizeof($temp); $z++)
					$cols[$z] = $temp[$z][1];
				if($fields_in_first_row)
				{
					for($z=0; $z<sizeof($cols); $z++)
					{
						echo $field_enclosed.$cols[$z].$field_enclosed;
						// do not terminate the last column!
						if($z < sizeof($cols)-1)
							echo $field_terminate;
					}
					echo "\r\n";	
				}
				$query = "SELECT * FROM ".$result[$i]['tbl_name'];
				$arr = $this->selectArray($query, "assoc");
				for($z=0; $z<sizeof($arr); $z++)
				{
					for($y=0; $y<sizeof($cols); $y++)
					{
						$cell = $arr[$z][$cols[$y]];
						if($crlf)
						{
							$cell = str_replace("\n","", $cell);
							$cell = str_replace("\r","", $cell);
						}
						$cell = str_replace($field_terminate,$field_escaped.$field_terminate,$cell);
						$cell = str_replace($field_enclosed,$field_escaped.$field_enclosed,$cell);
						// do not enclose NULLs
						if($cell == NULL)
							echo $null;  
						else
							echo $field_enclosed.$cell.$field_enclosed;
						// do not terminate the last column!
						if($y < sizeof($cols)-1)
							echo $field_terminate;
					}
					if($z<sizeof($arr)-1)
						echo "\r\n";	
				}
				if($i<sizeof($result)-1)
					echo "\r\n";
			}
		}
	}
	
	//export sql
	public function export_sql($tables, $drop, $structure, $data, $transaction, $comments)
	{
		if($comments)
		{
			echo "----\r\n";
			echo "-- phpLiteAdmin database dump (http://phpliteadmin.googlecode.com)\r\n";
			echo "-- phpLiteAdmin version: ".VERSION."\r\n";
			echo "-- Exported on ".date('M jS, Y, h:i:sA')."\r\n";
			echo "-- Database file: ".$this->getPath()."\r\n";
			echo "----\r\n";
		}
		$query = "SELECT * FROM sqlite_master WHERE type='table' OR type='index' ORDER BY type DESC";
		$result = $this->selectArray($query);

		//iterate through each table
		for($i=0; $i<sizeof($result); $i++)
		{
			$valid = false;
			for($j=0; $j<sizeof($tables); $j++)
			{
				if($result[$i]['tbl_name']==$tables[$j])
					$valid = true;
			}
			if($valid)
			{
				if($drop)
				{
					if($comments)
					{
						echo "\r\n----\r\n";
						if($result[$i]['type']=="table")
							echo "-- Drop table for ".$result[$i]['tbl_name']."\r\n";
						else
							echo "-- Drop index for ".$result[$i]['name']."\r\n";
						echo "----\r\n";
					}
					if($result[$i]['type']=="table")
						echo "DROP TABLE '".$result[$i]['tbl_name']."';\r\n";
					else
						echo "DROP INDEX '".$result[$i]['name']."';\r\n";
				}
				if($structure)
				{
					if($comments)
					{
						echo "\r\n----\r\n";
						if($result[$i]['type']=="table")
							echo "-- Table structure for ".$result[$i]['tbl_name']."\r\n";
						else
							echo "-- Structure for index ".$result[$i]['name']." on table ".$result[$i]['tbl_name']."\r\n";
						echo "----\r\n";
					}
					echo $result[$i]['sql'].";\r\n";
				}
				if($data && $result[$i]['type']=="table")
				{
					$query = "SELECT * FROM ".$result[$i]['tbl_name'];
					$arr = $this->selectArray($query, "assoc");

					if($comments)
					{
						echo "\r\n----\r\n";
						echo "-- Data dump for ".$result[$i]['tbl_name'].", a total of ".sizeof($arr)." rows\r\n";
						echo "----\r\n";
					}
					$query = "PRAGMA table_info('".$result[$i]['tbl_name']."')";
					$temp = $this->selectArray($query);
					$cols = array();
					$vals = array();
					for($z=0; $z<sizeof($temp); $z++)
						$cols[$z] = $temp[$z][1];
					for($z=0; $z<sizeof($arr); $z++)
					{
						for($y=0; $y<sizeof($cols); $y++)
						{
							if(!isset($vals[$z]))
								$vals[$z] = array();
							$vals[$z][$cols[$y]] = $this->quote($arr[$z][$cols[$y]]);
						}
					}
					if($transaction)
						echo "BEGIN TRANSACTION;\r\n";
					for($j=0; $j<sizeof($vals); $j++)
						echo "INSERT INTO ".$result[$i]['tbl_name']." (".implode(",", $cols).") VALUES (".implode(",", $vals[$j]).");\r\n";
					if($transaction)
						echo "COMMIT;\r\n";
				}
			}
		}
	}
}

$auth = new Authorization(); //create authorization object
if(isset($_POST['logout'])) //user has attempted to log out
	$auth->revoke();
else if(isset($_POST['login']) || isset($_POST['proc_login'])) //user has attempted to log in
{
	$_POST['login'] = true;

	if($_POST['password']==SYSTEMPASSWORD) //make sure passwords match before granting authorization
	{
		if(isset($_POST['remember']))
			$auth->grant(true);
		else
			$auth->grant(false);
	}
}

//user is downloading the exported database file
if(isset($_POST['export']))
{
	if($_POST['export_type']=="sql")
	{
		header('Content-Type: text/sql');
		header('Content-Disposition: attachment; filename="'.$_POST['filename'].'.'.$_POST['export_type'].'";');
		if(isset($_POST['tables']))
			$tables = $_POST['tables'];
		else
		{
			$tables = array();
			$tables[0] = $_POST['single_table'];
		}
		$drop = isset($_POST['drop']);
		$structure = isset($_POST['structure']);
		$data = isset($_POST['data']);
		$transaction = isset($_POST['transaction']);
		$comments = isset($_POST['comments']);
		$db = new Database($databases[$_SESSION[COOKIENAME.'currentDB']]);
		echo $db->export_sql($tables, $drop, $structure, $data, $transaction, $comments);
	}
	else if($_POST['export_type']=="csv")
	{
		header("Content-type: application/csv");
		header('Content-Disposition: attachment; filename="'.$_POST['filename'].'.'.$_POST['export_type'].'";');
		header("Pragma: no-cache");
		header("Expires: 0");
		if(isset($_POST['tables']))
			$tables = $_POST['tables'];
		else
		{
			$tables = array();
			$tables[0] = $_POST['single_table'];
		}
		$field_terminate = $_POST['export_csv_fieldsterminated'];
		$field_enclosed = $_POST['export_csv_fieldsenclosed'];
		$field_escaped = $_POST['export_csv_fieldsescaped'];
		$null = $_POST['export_csv_replacenull'];
		$crlf = isset($_POST['export_csv_crlf']);
		$fields_in_first_row = isset($_POST['export_csv_fieldnames']);
		$db = new Database($databases[$_SESSION[COOKIENAME.'currentDB']]);
		echo $db->export_csv($tables, $field_terminate, $field_enclosed, $field_escaped, $null, $crlf, $fields_in_first_row);
	}
	exit();
}

//user is importing a file
if(isset($_POST['import']))
{
	$db = new Database($databases[$_SESSION[COOKIENAME.'currentDB']]);
	if($_POST['import_type']=="sql")
	{
		$data = file_get_contents($_FILES["file"]["tmp_name"]);
		$importSuccess = $db->import_sql($data);
	}
	else
	{
		$field_terminate = $_POST['import_csv_fieldsterminated'];
		$field_enclosed = $_POST['import_csv_fieldsenclosed'];
		$field_escaped = $_POST['import_csv_fieldsescaped'];
		$null = $_POST['import_csv_replacenull'];
		$fields_in_first_row = isset($_POST['import_csv_fieldnames']);
		$importSuccess = $db->import_csv($_FILES["file"]["tmp_name"], $_POST['single_table'], $field_terminate, $field_enclosed, $field_escaped, $null, $fields_in_first_row);
	}
}

// here begins the HTML.
?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<!-- Copyright 2011 phpLiteAdmin (http://phpliteadmin.googlecode.com) -->
<meta http-equiv='Content-Type' content='text/html; charset=UTF-8' />
<title><?php echo PROJECT ?></title>

<?php
if(!file_exists("phpliteadmin.css")) //only use the inline stylesheet if an external one does not exist
{
?>
<!-- begin the customizable stylesheet/theme -->
<style type="text/css">
/* overall styles for entire page */
body
{
	margin: 0px;
	padding: 0px;
	font-family: Arial, Helvetica, sans-serif;
	font-size: 14px;
	color: #000000;
	background-color: #e0ebf6;
}
/* general styles for hyperlink */
a
{
	color: #03F;
	text-decoration: none;
	cursor :pointer;
}
hr
{
	height: 1px;
	border: 0;
	color: #bbb;
	background-color: #bbb;
	width: 100%;	
}
a:hover
{
	color: #06F;
}
/* logo text containing name of project */
h1
{
	margin: 0px;
	padding: 5px;
	font-size: 24px;
	background-color: #f3cece;
	text-align: center;
	color: #000;
	border-top-left-radius:5px;
	border-top-right-radius:5px;
	-moz-border-radius-topleft:5px;
	-moz-border-radius-topright:5px;
}
/* the div container for the links */
#headerlinks
{
	text-align:center;
	margin-bottom:10px;
	padding:5px;
	border-color:#03F;
	border-width:1px;
	border-style:solid;
	border-left-style:none;
	border-right-style:none;
	font-size:12px;
	background-color:#e0ebf6;
	font-weight:bold;
}
/* version text within the logo */
h1 #version
{
	color: #000000;
	font-size: 16px;
}
/* logo text within logo */
h1 #logo
{
	color:#000;
}
/* general header for various views */
h2
{
	margin:0px;
	padding:0px;
	font-size:14px;
	margin-bottom:20px;
}
/* input buttons and areas for entering text */
input, select, textarea
{
	font-family:Arial, Helvetica, sans-serif;
	background-color:#eaeaea;
	color:#03F;
	border-color:#03F;
	border-style:solid;
	border-width:1px;
	margin:5px;
	border-radius:5px;
	-moz-border-radius:5px;
	padding:3px;
}
/* just input buttons */
input.btn
{
	cursor:pointer;	
}
input.btn:hover
{
	background-color:#ccc;
}
/* general styles for hyperlink */
fieldset
{
	padding:15px;
	border-color:#03F;
	border-width:1px;
	border-style:solid;
	border-radius:5px;
	-moz-border-radius:5px;
	background-color:#f9f9f9;
}
/* outer div that holds everything */
#container
{
	padding:10px;
}
/* div of left box with log, list of databases, etc. */
#leftNav
{
	float:left;
	min-width:250px;
	padding:0px;
	border-color:#03F;
	border-width:1px;
	border-style:solid;
	background-color:#FFF;
	padding-bottom:15px;
	border-radius:5px;
	-moz-border-radius:5px;
}
/* div holding the content to the right of the leftNav */
#content
{
	overflow:hidden;
	padding-left:10px;
}
/* div holding the login fields */
#loginBox
{
	width:500px;
	margin-left:auto;
	margin-right:auto;
	margin-top:50px;
	border-color:#03F;
	border-width:1px;
	border-style:solid;
	background-color:#FFF;
	border-radius:5px;
	-moz-border-radius:5px;
}
/* div under tabs with tab-specific content */
#main
{
	border-color:#03F;
	border-width:1px;
	border-style:solid;
	padding:15px;
	overflow:auto;
	background-color:#FFF;
	border-bottom-left-radius:5px;
	border-bottom-right-radius:5px;
	border-top-right-radius:5px;
	-moz-border-radius-bottomleft:5px;
	-moz-border-radius-bottomright:5px;
	-moz-border-radius-topright:5px;
}
/* odd-numbered table rows */
.td1
{
	background-color:#f9e3e3;
	text-align:right;
	font-size:12px;
	padding-left:10px;
	padding-right:10px;
}
/* even-numbered table rows */
.td2
{
	background-color:#f3cece;
	text-align:right;
	font-size:12px;
	padding-left:10px;
	padding-right:10px;
}
/* table column headers */
.tdheader
{
	border-color:#03F;
	border-width:1px;
	border-style:solid;
	font-weight:bold;
	font-size:12px;
	padding-left:10px;
	padding-right:10px;
	background-color:#e0ebf6;
	border-radius:5px;
	-moz-border-radius:5px;
}
/* div holding the confirmation text of certain actions */
.confirm
{
	border-color:#03F;
	border-width:1px;
	border-style:dashed;
	padding:15px;
	background-color:#e0ebf6;
}
/* tab navigation for each table */
.tab
{
	display:block;
	padding:5px;
	padding-right:8px;
	padding-left:8px;
	border-color:#03F;
	border-width:1px;
	border-style:solid;
	margin-right:5px;
	float:left;
	border-bottom-style:none;
	position:relative;
	top:1px;
	padding-bottom:4px;
	background-color:#eaeaea;
	border-top-left-radius:5px;
	border-top-right-radius:5px;
	-moz-border-radius-topleft:5px;
	-moz-border-radius-topright:5px;
}
/* pressed state of tab */
.tab_pressed
{
	display:block;
	padding:5px;
	padding-right:8px;
	padding-left:8px;
	border-color:#03F;
	border-width:1px;
	border-style:solid;
	margin-right:5px;
	float:left;
	border-bottom-style:none;
	position:relative;
	top:1px;
	background-color:#FFF;
	cursor:default;
	border-top-left-radius:5px;
	border-top-right-radius:5px;
	-moz-border-radius-topleft:5px;
	-moz-border-radius-topright:5px;
}
/* help section */
.helpq
{
	font-size:11px;
	font-weight:normal;	
}
#help_container
{
	padding:0px;
	font-size:12px;
	margin-left:auto;
	margin-right:auto;
	background-color:#ffffff;
}
.help_outer
{
	background-color:#FFF;
	padding:0px;
	height:300px;
	overflow:hidden;
	position:relative;
}
.help_list
{
	padding:10px;
	height:auto;	
}

.headd
{
	font-size:14px;
	font-weight:bold;	
	display:block;
	padding:10px;
	background-color:#e0ebf6;
	border-color:#03F;
	border-width:1px;
	border-style:solid;
	border-left-style:none;
	border-right-style:none;
}
.help_inner
{
	padding:10px;	
}
.help_top
{
	display:block;
	position:absolute;
	right:10px;
	bottom:10px;	
}
</style>
<!-- end the customizable stylesheet/theme -->
<?php
}
else //an external stylesheet exists - import it
{
	echo 	"<link href='phpliteadmin.css' rel='stylesheet' type='text/css' />";
}
if(isset($_GET['help'])) //this page is used as the popup help section
{
	//help section array
	$help = array
	(
		'SQLite Library Extensions' => 
			'phpLiteAdmin uses PHP library extensions that allow interaction with SQLite databases. Currently, phpLiteAdmin supports PDO, SQLite3, and SQLiteDatabase. Both PDO and SQLite3 deal with version 3 of SQLite, while SQLiteDatabase deals with version 2. So, if your PHP installation includes more than one SQLite library extension, PDO and SQLite3 will take precedence to make use of the better technology. However, if you have existing databases that are of version 2 of SQLite, phpLiteAdmin will be forced to use SQLiteDatabase for only those databases. Not all databases need to be of the same version. During the database creation, however, the most advanced extension will be used.',
		'Creating a New Database' => 
			'When you create a new database, the name you entered will be appended with the appropriate file extension (.db, .db3, .sqlite, etc.) if you do not include it yourself. The database will be created in the directory you specified as the $directory variable.',
		'Tables vs. Views' => 
			'On the main database page, there is a list of tables and views. Since views are read-only, certain operations will be disabled. These disabled operations will be apparent by their omission in the location where they should appear on the row for a view. If you want to change the data for a view, you need to drop that view and create a new view with the appropriate SELECT statement that queries other existing tables. For more information, see <a href="http://en.wikipedia.org/wiki/View_(database)" target="_blank">http://en.wikipedia.org/wiki/View_(database)</a>',
		'Writing a Select Statement for a New View' => 
			'When you create a new view, you must write an SQL SELECT statement that it will use as its data. A view is simply a read-only table that can be accessed and queried like a regular table, except it cannot be modified through insertion, column editing, or row editing. It is only used for conveniently fetching data.',
		'Export Structure to SQL File' => 
			'During the process for exporting to an SQL file, you may choose to include the queries that create the table and columns.',
		'Export Data to SQL File' => 
			'During the process for exporting to an SQL file, you may choose to include the queries that populate the table(s) with the current records of the table(s).',
		'Add Drop Table to Exported SQL File' => 
			'During the process for exporting to an SQL file, you may choose to include queries to DROP the existing tables before adding them so that problems do not occur when trying to create tables that already exist.',
		'Add Transaction to Exported SQL File' => 
			'During the process for exporting to an SQL file, you may choose to wrap the queries around a TRANSACTION so that if an error occurs at any time during the importation process using the exported file, the database can be reverted to its previous state, preventing partially updated data from populating the database.',
		'Add Comments to Exported SQL File' => 
			'During the process for exporting to an SQL file, you may choose to include comments that explain each step of the process so that a human can better understand what is happening.',
	);
	?>
	</head>
	<body>
	<div id='help_container'>
	<?php
	echo "<div class='help_list'>";
	echo "<span style='font-size:18px;'>".PROJECT." v".VERSION." Help Documentation</span><br/><br/>";
	foreach((array)$help as $key => $val)
	{
		echo "<a href='#".$key."'>".$key."</a><br/>";
	}
	echo "</div>";
	echo "<br/><br/>";
	foreach((array)$help as $key => $val)
	{
		echo "<div class='help_outer'>";
		echo "<a class='headd' name='".$key."'>".$key."</a>";
		echo "<div class='help_inner'>";
		echo $val;
		echo "</div>";
		echo "<a class='help_top' href='#top'>Back to Top</a>";
		echo "</div>";
	}
	?>
	</div>
	</body>
	</html>
	<?php
	exit();		
}
?>
<!-- JavaScript Support -->
<script type="text/javascript">
//makes sure autoincrement can only be selected when integer type is selected
function toggleAutoincrement(i)
{
	var type = document.getElementById(i+'_type');
	var autoincrement = document.getElementById(i+'_autoincrement');
	if(type.value=="INTEGER")
		autoincrement.disabled = false;
	else
	{
		autoincrement.disabled = true;
		autoincrement.checked = false;
	}
}
function toggleNull(i)
{
	var pk = document.getElementById(i+'_primarykey');
	var notnull = document.getElementById(i+'_notnull');
	if(pk.checked)
	{
		notnull.disabled = true;
		notnull.checked = true;
	}
	else
	{
		notnull.disabled = false;
	}
}
//finds and checks all checkboxes for all rows on the Browse or Structure tab for a table
function checkAll(field)
{
	var i=0;
	while(document.getElementById('check_'+i)!=undefined)
	{
		document.getElementById('check_'+i).checked = true;
		i++;
	}
}
//finds and unchecks all checkboxes for all rows on the Browse or Structure tab for a table
function uncheckAll(field)
{
	var i=0;
	while(document.getElementById('check_'+i)!=undefined)
	{
		document.getElementById('check_'+i).checked = false;
		i++;
	}
}
//unchecks the ignore checkbox if user has typed something into one of the fields for adding new rows
function changeIgnore(area, e, u)
{
	if(area.value!="")
	{
		if(document.getElementById(e)!=undefined)
			document.getElementById(e).checked = false;
		if(document.getElementById(u)!=undefined)
			document.getElementById(u).checked = false;
	}
}
//moves fields from select menu into query textarea for SQL tab
function moveFields()
{
	var fields = document.getElementById("fieldcontainer");
	var selected = new Array();
	for(var i=0; i<fields.options.length; i++)
		if(fields.options[i].selected)
			selected.push(fields.options[i].value);
	for(var i=0; i<selected.length; i++)
		insertAtCaret("queryval", selected[i]);
}
//helper function for moveFields
function insertAtCaret(areaId,text)
{
	var txtarea = document.getElementById(areaId);
	var scrollPos = txtarea.scrollTop;
	var strPos = 0;
	var br = ((txtarea.selectionStart || txtarea.selectionStart == '0') ? "ff" : (document.selection ? "ie" : false ));
	if(br=="ie")
	{
		txtarea.focus();
		var range = document.selection.createRange();
		range.moveStart ('character', -txtarea.value.length);
		strPos = range.text.length;
	}
	else if(br=="ff")
		strPos = txtarea.selectionStart;

	var front = (txtarea.value).substring(0,strPos);
	var back = (txtarea.value).substring(strPos,txtarea.value.length);
	txtarea.value=front+text+back;
	strPos = strPos + text.length;
	if(br=="ie")
	{
		txtarea.focus();
		var range = document.selection.createRange();
		range.moveStart ('character', -txtarea.value.length);
		range.moveStart ('character', strPos);
		range.moveEnd ('character', 0);
		range.select();
	}
	else if(br=="ff")
	{
		txtarea.selectionStart = strPos;
		txtarea.selectionEnd = strPos;
		txtarea.focus();
	}
	txtarea.scrollTop = scrollPos;
}

function notNull(checker)
{
	document.getElementById(checker).checked = false;
}

function disableText(checker, textie)
{
	if(checker.checked)
	{
		document.getElementById(textie).disabled = true;	
	}
	else
	{
		document.getElementById(textie).disabled = false;	
	}
}

function toggleExports(val)
{
	document.getElementById("exportoptions_sql").style.display = "none";	
	document.getElementById("exportoptions_csv").style.display = "none";	
	
	document.getElementById("exportoptions_"+val).style.display = "block";	
}

function toggleImports(val)
{
	document.getElementById("importoptions_sql").style.display = "none";	
	document.getElementById("importoptions_csv").style.display = "none";	
	
	document.getElementById("importoptions_"+val).style.display = "block";	
}

function openHelp(section)
{
	PopupCenter('<?php echo PAGE."?help=1"; ?>#'+section, "Help Section");	
}
var helpsec = false;
function PopupCenter(pageURL, title)
{
	helpsec = window.open(pageURL, title, "toolbar=0,scrollbars=1,location=0,statusbar=0,menubar=0,resizable=0,width=400,height=300");
} 

</script>
</head>
<body>
<?php
if(ini_get("register_globals")) //check whether register_globals is turned on - if it is, we need to not continue
{
	echo "<div class='confirm' style='margin:20px;'>";
	echo "It appears that the PHP directive, 'register_globals' is enabled. This is bad. You need to disable it before continuing.";
	echo "</div>";
	exit();
}

if(!$auth->isAuthorized()) //user is not authorized - display the login screen
{
	echo "<div id='loginBox'>";
	echo "<h1><span id='logo'>".PROJECT."</span> <span id='version'>v".VERSION."</span></h1>";
	echo "<div style='padding:15px; text-align:center;'>";
	if(isset($_POST['login']))
		echo "<span style='color:red;'>Incorrect password.</span><br/><br/>";
	echo "<form action='".PAGE."' method='post'>";
	echo "Password: <input type='password' name='password'/><br/>";
	echo "<input type='checkbox' name='remember' value='yes' checked='checked'/> Remember me<br/><br/>";
	echo "<input type='submit' value='Log In' name='login' class='btn'/>";
	echo "<input type='hidden' name='proc_login' value='true' />";
	echo "</form>";
	echo "</div>";
	echo "</div>";
	echo "<br/>";
	echo "<div style='text-align:center;'>";
	$endTimeTot = microtime(true);
	$timeTot = round(($endTimeTot - $startTimeTot), 4);
	echo "<span style='font-size:11px;'>Powered by <a href='http://phpliteadmin.googlecode.com' target='_blank' style='font-size:11px;'>".PROJECT."</a> | Page generated in ".$timeTot." seconds.</span>";
	echo "</div>";
}
else //user is authorized - display the main application
{
	if(!isset($_SESSION[COOKIENAME.'currentDB']))
		$_SESSION[COOKIENAME.'currentDB'] = 0;
	//set the current database to the first in the array (default)
	if(sizeof($databases)>0)
		$currentDB = $databases[0];
	else //the database array is empty - show error and halt execution
	{
		if($directory!==false && is_writable($directory))
		{
			echo "<div class='confirm' style='margin:20px;'>";
			echo "Welcome to phpLiteAdmin. It appears that you have selected to scan a directory for databases to manage. However, phpLiteAdmin could not find any valid SQLite databases. You may use the form below to create your first database.";
			echo "</div>";	
			echo "<fieldset style='margin:15px;'><legend><b>Create New Database</b></legend>";
			echo "<form name='create_database' method='post' action='".$_SERVER['PHP_SELF']."'>";
			echo "<input type='text' name='new_dbname' style='width:150px;'/> <input type='submit' value='Create' class='btn'/>";
			echo "</form>";
			echo "</fieldset>";
		}
		else
		{
			echo "<div class='confirm' style='margin:20px;'>";
			echo "Error: The directory you specified does not contain any existing databases to manage, and the directory is not writable. This means you can't create any new databases using phpLiteAdmin. Either make the directory writable or manually upload databases to the directory.";
			echo "</div><br/>";	
		}
		exit();
	}

	if(isset($_POST['database_switch'])) //user is switching database with drop-down menu
	{
		$_SESSION[COOKIENAME."currentDB"] = $_POST['database_switch'];
		$currentDB = $databases[$_SESSION[COOKIENAME.'currentDB']];
	}
	else if(isset($_GET['switchdb']))
	{
		$_SESSION[COOKIENAME."currentDB"] = $_GET['switchdb'];
		$currentDB = $databases[$_SESSION[COOKIENAME.'currentDB']];
	}
	if(isset($_SESSION[COOKIENAME.'currentDB']))
		$currentDB = $databases[$_SESSION[COOKIENAME.'currentDB']];

	//create the objects
	$db = new Database($currentDB); //create the Database object

	//switch board for various operations a user could have requested - these actions are invisible and produce no output
	if(isset($_GET['action']) && isset($_GET['confirm']))
	{
		switch($_GET['action'])
		{
			//table actions
			/////////////////////////////////////////////// create table
			case "table_create":
				$num = intval($_POST['rows']);
				$name = $_POST['tablename'];
				$query = "CREATE TABLE ".$name."(";
				for($i=0; $i<$num; $i++)
				{
					if($_POST[$i.'_field']!="")
					{
						$query .= $_POST[$i.'_field']." ";
						$query .= $_POST[$i.'_type']." ";
						if(isset($_POST[$i.'_primarykey']))
							$query .= "PRIMARY KEY NOT NULL ";
						if(!isset($_POST[$i.'_primarykey']) && isset($_POST[$i.'_notnull']))
							$query .= "NOT NULL ";
						if($_POST[$i.'_defaultvalue']!="")
						{
							if($_POST[$i.'_type']=="INTEGER")
								$query .= "default ".$_POST[$i.'_defaultvalue']."  ";
							else
								$query .= "default '".$_POST[$i.'_defaultvalue']."' ";
						}
						$query = substr($query, 0, sizeof($query)-2);
						$query .= ", ";
					}
				}
				$query = substr($query, 0, sizeof($query)-3);
				$query .= ")";
				$result = $db->query($query);
				if(!$result)
					$error = true;
				$completed = "Table '".$_POST['tablename']."' has been created.<br/><span style='font-size:11px;'>".$query."</span>";
				break;
			/////////////////////////////////////////////// empty table
			case "table_empty":
				$query = "DELETE FROM ".$_POST['tablename'];
				$result = $db->query($query);
				if(!$result)
					$error = true;
				$query = "VACUUM";
				$result = $db->query($query);
				if(!$result)
					$error = true;
				$completed = "Table '".$_POST['tablename']."' has been emptied.<br/><span style='font-size:11px;'>".$query."</span>";
				break;
			/////////////////////////////////////////////// create view
			case "view_create":
				$query = "CREATE VIEW ".$_POST['viewname']." AS ".stripslashes($_POST['select']);
				$result = $db->query($query);
				if(!$result)
					$error = true;
				$completed = "View '".$_POST['viewname']."' has been created.<br/><span style='font-size:11px;'>".$query."</span>";
				break;
			/////////////////////////////////////////////// drop table
			case "table_drop":
				$query = "DROP TABLE ".$_POST['tablename'];
				$db->query($query);
				$completed = "Table '".$_POST['tablename']."' has been dropped.";
				break;
			/////////////////////////////////////////////// drop view
			case "view_drop":
				$query = "DROP VIEW ".$_POST['viewname'];
				$db->query($query);
				$completed = "View '".$_POST['viewname']."' has been dropped.";
				break;
			/////////////////////////////////////////////// rename table
			case "table_rename":
				$query = "ALTER TABLE ".$_POST['oldname']." RENAME TO ".$_POST['newname'];
				if($db->getVersion()==3)
					$result = $db->query($query, true);
				else
					$result = $db->query($query, false);
				if(!$result)
					$error = true;
				$completed = "Table '".$_POST['oldname']."' has been renamed to '".$_POST['newname']."'.<br/><span style='font-size:11px;'>".$query."</span>";
				break;
			//row actions
			/////////////////////////////////////////////// create row
			case "row_create":
				$completed = "";
				$num = $_POST['numRows'];
				$fields = explode(":", $_POST['fields']);
				$z = 0;
				
				$query = "PRAGMA table_info('".$_GET['table']."')";
				$result = $db->selectArray($query);
				
				for($i=0; $i<$num; $i++)
				{
					if(!isset($_POST[$i.":ignore"]))
					{
						$query = "INSERT INTO ".$_GET['table']." (";
						for($j=0; $j<sizeof($fields); $j++)
						{
							$query .= $fields[$j].",";
						}
						$query = substr($query, 0, sizeof($query)-2);
						$query .= ") VALUES (";
						for($j=0; $j<sizeof($fields); $j++)
						{
							$value = $_POST[$i.":".$fields[$j]];
							$null = isset($_POST[$i.":".$fields[$j]."_null"]);
							$type = $result[$j][2];
							$function = $_POST["function_".$i."_".$fields[$j]];
							if($function!="")
								$query .= $function."(";
								//di - messed around with this logic for null values
							if(($type=="TEXT" || $type=="BLOB") && $null==false)
								$query .= $db->quote($value);
							else if(($type=="INTEGER" || $type=="REAL") && $null==false && $value=="")
								$query .= "NULL";
							else if($null==true)
								$query .= "NULL";
							else
								$query .= $db->quote($value);
							if($function!="")
								$query .= ")";
							$query .= ",";
						}
						$query = substr($query, 0, sizeof($query)-2);
						$query .= ")";
						$result1 = $db->query($query);
						if(!$result1)
							$error = true;
						$completed .= "<span style='font-size:11px;'>".$query."</span><br/>";
						$z++;
					}
				}
				$completed = $z." row(s) inserted.<br/><br/>".$completed;
				break;
			/////////////////////////////////////////////// delete row
			case "row_delete":
				$pks = explode(":", $_GET['pk']);
				$str = $pks[0];
				$query = "DELETE FROM ".$_GET['table']." WHERE ROWID = ".$pks[0];
				for($i=1; $i<sizeof($pks); $i++)
				{
					$str .= ", ".$pks[$i];
					$query .= " OR ROWID = ".$pks[$i];
				}
				$result = $db->query($query);
				if(!$result)
					$error = true;
				$completed = sizeof($pks)." row(s) deleted.<br/><span style='font-size:11px;'>".$query."</span>";
				break;
			/////////////////////////////////////////////// edit row
			case "row_edit":
				$pks = explode(":", $_GET['pk']);
				$fields = explode(":", $_POST['fieldArray']);
				
				$z = 0;
				
				$query = "PRAGMA table_info('".$_GET['table']."')";
				$result = $db->selectArray($query);
				
				if(isset($_POST['new_row']))
					$completed = "";
				else
					$completed = sizeof($pks)." row(s) affected.<br/><br/>";

				for($i=0; $i<sizeof($pks); $i++)
				{
					if(isset($_POST['new_row']))
					{
						$query = "INSERT INTO ".$_GET['table']." (";
						for($j=0; $j<sizeof($fields); $j++)
						{
							$query .= $fields[$j].",";
						}
						$query = substr($query, 0, sizeof($query)-2);
						$query .= ") VALUES (";
						for($j=0; $j<sizeof($fields); $j++)
						{
							$value = $_POST[$pks[$i].":".$fields[$j]];
							$null = isset($_POST[$pks[$i].":".$fields[$j]."_null"]);
							$type = $result[$j][2];
							$function = $_POST["function_".$pks[$i]."_".$fields[$j]];
							if($function!="")
								$query .= $function."(";
								//di - messed around with this logic for null values
							if(($type=="TEXT" || $type=="BLOB") && $null==false)
								$query .= $db->quote($value);
							else if(($type=="INTEGER" || $type=="REAL") && $null==false && $value=="")
								$query .= "NULL";
							else if($null==true)
								$query .= "NULL";
							else
								$query .= $db->quote($value);
							if($function!="")
								$query .= ")";
							$query .= ",";
						}
						$query = substr($query, 0, sizeof($query)-2);
						$query .= ")";
						$result1 = $db->query($query);
						if(!$result1)
							$error = true;
						$z++;
					}
					else
					{
						$query = "UPDATE ".$_GET['table']." SET ";
						for($j=0; $j<sizeof($fields); $j++)
						{
							$function = $_POST["function_".$pks[$i]."_".$fields[$j]];
							$null = isset($_POST[$pks[$i].":".$fields[$j]."_null"]);
							$query .= $fields[$j]."=";
							if($function!="")
								$query .= $function."(";
							if($null)
								$query .= "NULL";
							else
								$query .= $db->quote($_POST[$pks[$i].":".$fields[$j]]);
							if($function!="")
								$query .= ")";
							$query .= ", ";
						}
						$query = substr($query, 0, sizeof($query)-3);
						$query .= " WHERE ROWID = ".$pks[$i];
						$result1 = $db->query($query);
						if(!$result1)
						{
							$error = true;
						}
					}
					$completed .= "<span style='font-size:11px;'>".$query."</span><br/>";
				}
				if(isset($_POST['new_row']))
					$completed = $z." row(s) inserted.<br/><br/>".$completed;
				break;
			//column actions
			/////////////////////////////////////////////// create column
			case "column_create":
				$num = intval($_POST['rows']);
				for($i=0; $i<$num; $i++)
				{
					if($_POST[$i.'_field']!="")
					{
						$query = "ALTER TABLE ".$_GET['table']." ADD ".$_POST[$i.'_field']." ";
						$query .= $_POST[$i.'_type']." ";
						if(isset($_POST[$i.'_primarykey']))
							$query .= "PRIMARY KEY ";
						if(isset($_POST[$i.'_notnull']))
							$query .= "NOT NULL ";
						if($_POST[$i.'_defaultvalue']!="")
						{
							if($_POST[$i.'_type']=="INTEGER")
								$query .= "DEFAULT ".$_POST[$i.'_defaultvalue']."  ";
							else
								$query .= "DEFAULT '".$_POST[$i.'_defaultvalue']."' ";
						}
						if($db->getVersion()==3)
							$result = $db->query($query, true);
						else
							$result = $db->query($query, false);
						if(!$result)
							$error = true;
					}
				}
				$completed = "Table '".$_GET['table']."' has been altered successfully.";
				break;
			/////////////////////////////////////////////// delete column
			case "column_delete":
				$pks = explode(":", $_GET['pk']);
				$str = $pks[0];
				$query = "ALTER TABLE ".$_GET['table']." DROP ".$pks[0];
				for($i=1; $i<sizeof($pks); $i++)
				{
					$str .= ", ".$pks[$i];
					$query .= ", DROP ".$pks[$i];
				}
				$result = $db->query($query);
				if(!$result)
					$error = true;
				$completed = "Table '".$_GET['table']."' has been altered successfully.";
				break;
			/////////////////////////////////////////////// edit column
			case "column_edit":
				$query = "ALTER TABLE ".$_GET['table']." CHANGE ".$_POST['oldvalue']." ".$_POST['0_field']." ".$_POST['0_type'];
				$result = $db->query($query);
				if(!$result)
					$error = true;
				$completed = "Table '".$_GET['table']."' has been altered successfully.";
				break;
			/////////////////////////////////////////////// delete trigger
			case "trigger_delete":
				$query = "DROP TRIGGER ".$_GET['pk'];
				$result = $db->query($query);
				if(!$result)
					$error = true;
				$completed = "Trigger '".$_GET['pk']."' deleted.<br/><span style='font-size:11px;'>".$query."</span>";
				break;
			/////////////////////////////////////////////// delete index
			case "index_delete":
				$query = "DROP INDEX ".$_GET['pk'];
				$result = $db->query($query);
				if(!$result)
					$error = true;
				$completed = "Index '".$_GET['pk']."' deleted.<br/><span style='font-size:11px;'>".$query."</span>";
				break;
			/////////////////////////////////////////////// create trigger
			case "trigger_create":
				$str = "CREATE TRIGGER ".$_POST['trigger_name'];
				if($_POST['beforeafter']!="")
					$str .= " ".$_POST['beforeafter'];
				$str .= " ".$_POST['event']." ON ".$_GET['table'];
				if(isset($_POST['foreachrow']))
					$str .= " FOR EACH ROW";
				if($_POST['whenexpression']!="")
					$str .= " WHEN ".stripslashes($_POST['whenexpression']);
				$str .= " BEGIN";
				$str .= " ".stripslashes($_POST['triggersteps']);
				$str .= " END";
				$query = $str;
				$result = $db->query($query);
				if(!$result)
					$error = true;
				$completed = "Trigger created.<br/><span style='font-size:11px;'>".$query."</span>";
				break;
			/////////////////////////////////////////////// create index
			case "index_create":
				$num = $_POST['num'];
				if($_POST['name']=="")
				{
					$completed = "Index name must not be blank.";
				}
				else if($_POST['0_field']=="")
				{
					$completed = "You must specify at least one index column.";
				}
				else
				{
					$str = "CREATE ";
					if($_POST['duplicate']=="no")
						$str .= "UNIQUE ";
					$str .= "INDEX ".$_POST['name']." ON ".$_GET['table']." (";
					$str .= $_POST['0_field'].$_POST['0_order'];
					for($i=1; $i<$num; $i++)
					{
						if($_POST[$i.'_field']!="")
							$str .= ", ".$_POST[$i.'_field'].$_POST[$i.'_order'];
					}
					$str .= ")";
					$query = $str;
					$result = $db->query($query);
					if(!$result)
						$error = true;
					$completed = "Index created.<br/><span style='font-size:11px;'>".$query."</span>";
				}
				break;
		}
	}

	echo "<div id='container'>";
	echo "<div id='leftNav'>";
	echo "<h1>";
	echo "<a href='".PAGE."'>";
	echo "<span id='logo'>".PROJECT."</span> <span id='version'>v".VERSION."</span>";
	echo "</a>";
	echo "</h1>";
	echo "<div id='headerlinks'>";
	echo "<a href='javascript:openHelp(\"top\");'>Documentation</a> | ";
	echo "<a href='http://www.gnu.org/licenses/gpl.html' target='_blank'>License</a> | ";
	echo "<a href='http://code.google.com/p/phpliteadmin/' target='_blank'>Project Site</a>";
	echo "</div>";
	echo "<fieldset style='margin:15px;'><legend><b>Change Database</b></legend>";
	if(sizeof($databases)<10) //if there aren't a lot of databases, just show them as a list of links instead of drop down menu
	{
		for($i=0; $i<sizeof($databases); $i++)
		{
			// 22 August 2011: gkf fixed bug #49
			echo $databases[$i]['perms'];
			if($i==$_SESSION[COOKIENAME.'currentDB'])
				echo "<a href='".PAGE."?switchdb=".$i."' style='text-decoration:underline;'>".$databases[$i]['name']."</a>";
			else
				echo "<a href='".PAGE."?switchdb=".$i."'>".$databases[$i]['name']."</a>";
			if($i<sizeof($databases)-1)
				echo "<br/>";
		}
	}
	else //there are a lot of databases - show a drop down menu
	{
		echo "<form action='".PAGE."' method='post'>";
		echo "<select name='database_switch'>";
		// 22 August 2011: gkf fixed bug #49
		for($i=0; $i<sizeof($databases); $i++)
		{
			if($i==$_SESSION[COOKIENAME.'currentDB'])
				echo "<option value='".$i."' selected='selected'>".$databases[$i]['perms'].$databases[$i]['name']."</option>";
			else
				echo "<option value='".$i."'>".$databases[$i]['perms'].$databases[$i]['name']."</option>";
		}
		echo "</select> ";
		echo "<input type='submit' value='Go' class='btn'>";
		echo "</form>";
	}
	echo "</fieldset>";
	echo "<fieldset style='margin:15px;'><legend>";
	echo "<a href='".PAGE."'";
	if(!isset($_GET['table']))
		echo " style='text-decoration:underline;'";
	echo ">".$currentDB['name']."</a>";
	echo "</legend>";
	//Display list of tables
	$query = "SELECT type, name FROM sqlite_master WHERE type='table' OR type='view' ORDER BY name";
	$result = $db->selectArray($query);
	$j=0;
	for($i=0; $i<sizeof($result); $i++)
	{
		if(substr($result[$i]['name'], 0, 7)!="sqlite_" && $result[$i]['name']!="")
		{
			if($result[$i]['type']=="table")
				echo "<span style='font-size:11px;'>[table]</span> <a href='".PAGE."?action=row_view&table=".$result[$i]['name']."'";
			else
				echo "<span style='font-size:11px;'>[view]</span> <a href='".PAGE."?action=row_view&table=".$result[$i]['name']."&view=1'";
			if(isset($_GET['table']) && $_GET['table']==$result[$i]['name'])
				echo " style='text-decoration:underline;'";
			echo ">".$result[$i]['name']."</a><br/>";
			$j++;
		}
	}
	if($j==0)
		echo "No tables in database.";
	echo "</fieldset>";
	
	if($directory!==false && is_writable($directory))
	{
		echo "<fieldset style='margin:15px;'><legend><b>Create New Database</b> ".helpLink("Creating a New Database")."</legend>";
		echo "<form name='create_database' method='post' action='".$_SERVER['PHP_SELF']."'>";
		echo "<input type='text' name='new_dbname' style='width:150px;'/> <input type='submit' value='Create' class='btn'/>";
		echo "</form>";
		echo "</fieldset>";
	}
	
	echo "<div style='text-align:center;'>";
	echo "<form action='".PAGE."' method='post'/>";
	echo "<input type='submit' value='Log Out' name='logout' class='btn'/>";
	echo "</form>";
	echo "</div>";
	echo "</div>";
	echo "<div id='content'>";

	//breadcrumb navigation
	echo "<a href='".PAGE."'>".$currentDB['name']."</a>";
	if(isset($_GET['table']))
		echo " &rarr; <a href='".PAGE."?table=".$_GET['table']."&action=row_view'>".$_GET['table']."</a>";
	echo "<br/><br/>";

	//user has performed some action so show the resulting message
	if(isset($_GET['confirm']))
	{
		echo "<div id='main'>";
		echo "<div class='confirm'>";
		if(isset($error) && $error) //an error occured during the action, so show an error message
			echo "Error: ".$db->getError().".<br/>This may be a bug that needs to be reported at <a href='http://code.google.com/p/phpliteadmin/issues/list' target='_blank'>code.google.com/p/phpliteadmin/issues/list</a>";
		else //action was performed successfully - show success message
			echo $completed;
		echo "</div>";
		if($_GET['action']=="row_delete" || $_GET['action']=="row_create" || $_GET['action']=="row_edit")
			echo "<br/><br/><a href='".PAGE."?table=".$_GET['table']."&action=row_view'>Return</a>";
		else if($_GET['action']=="column_create" || $_GET['action']=="column_delete" || $_GET['action']=="column_edit" || $_GET['action']=="index_create" || $_GET['action']=="index_delete" || $_GET['action']=="trigger_delete" || $_GET['action']=="trigger_create")
			echo "<br/><br/><a href='".PAGE."?table=".$_GET['table']."&action=column_view'>Return</a>";
		else
			echo "<br/><br/><a href='".PAGE."'>Return</a>";
		echo "</div>";
	}

	//show the various tab views for a table
	if(!isset($_GET['confirm']) && isset($_GET['table']) && isset($_GET['action']) && ($_GET['action']=="table_export" || $_GET['action']=="table_import" || $_GET['action']=="table_sql" || $_GET['action']=="row_view" || $_GET['action']=="row_create" || $_GET['action']=="column_view" || $_GET['action']=="table_rename" || $_GET['action']=="table_search" || $_GET['action']=="table_triggers"))
	{
		if(!isset($_GET['view']))
		{
			echo "<a href='".PAGE."?table=".$_GET['table']."&action=row_view' ";
			if($_GET['action']=="row_view")
				echo "class='tab_pressed'";
			else
				echo "class='tab'";
			echo ">Browse</a>";
			echo "<a href='".PAGE."?table=".$_GET['table']."&action=column_view' ";
			if($_GET['action']=="column_view")
				echo "class='tab_pressed'";
			else
				echo "class='tab'";
			echo ">Structure</a>";
			echo "<a href='".PAGE."?table=".$_GET['table']."&action=table_sql' ";
			if($_GET['action']=="table_sql")
				echo "class='tab_pressed'";
			else
				echo "class='tab'";
			echo ">SQL</a>";
			echo "<a href='".PAGE."?table=".$_GET['table']."&action=table_search' ";
			if($_GET['action']=="table_search")
				echo "class='tab_pressed'";
			else
				echo "class='tab'";
			echo ">Search</a>";
			echo "<a href='".PAGE."?table=".$_GET['table']."&action=row_create' ";
			if($_GET['action']=="row_create")
				echo "class='tab_pressed'";
			else
				echo "class='tab'";
			echo ">Insert</a>";
			echo "<a href='".PAGE."?table=".$_GET['table']."&action=table_export' ";
			if($_GET['action']=="table_export")
				echo "class='tab_pressed'";
			else
				echo "class='tab'";
			echo ">Export</a>";
			echo "<a href='".PAGE."?table=".$_GET['table']."&action=table_import' ";
			if($_GET['action']=="table_import")
				echo "class='tab_pressed'";
			else
				echo "class='tab'";
			echo ">Import</a>";
			echo "<a href='".PAGE."?table=".$_GET['table']."&action=table_rename' ";
			if($_GET['action']=="table_rename")
				echo "class='tab_pressed'";
			else
				echo "class='tab'";
			echo ">Rename</a>";
			echo "<a href='".PAGE."?action=table_empty&table=".$_GET['table']."' ";
			echo "class='tab' style='color:red;'";
			echo ">Empty</a>";
			echo "<a href='".PAGE."?action=table_drop&table=".$_GET['table']."' ";
			echo "class='tab' style='color:red;'";
			echo ">Drop</a>";
			echo "<div style='clear:both;'></div>";
		}
		else
		{
			echo "<a href='".PAGE."?table=".$_GET['table']."&action=row_view&view=1' ";
			if($_GET['action']=="row_view")
				echo "class='tab_pressed'";
			else
				echo "class='tab'";
			echo ">Browse</a>";
			echo "<a href='".PAGE."?table=".$_GET['table']."&action=column_view&view=1' ";
			if($_GET['action']=="column_view")
				echo "class='tab_pressed'";
			else
				echo "class='tab'";
			echo ">Structure</a>";
			echo "<a href='".PAGE."?table=".$_GET['table']."&action=table_sql&view=1' ";
			if($_GET['action']=="table_sql")
				echo "class='tab_pressed'";
			else
				echo "class='tab'";
			echo ">SQL</a>";
			echo "<a href='".PAGE."?table=".$_GET['table']."&action=table_search&view=1' ";
			if($_GET['action']=="table_search")
				echo "class='tab_pressed'";
			else
				echo "class='tab'";
			echo ">Search</a>";
			echo "<a href='".PAGE."?table=".$_GET['table']."&action=table_export&view=1' ";
			if($_GET['action']=="table_export")
				echo "class='tab_pressed'";
			else
				echo "class='tab'";
			echo ">Export</a>";
			echo "<a href='".PAGE."?action=view_drop&table=".$_GET['table']."&view=1' ";
			echo "class='tab' style='color:red;'";
			echo ">Drop</a>";
			echo "<div style='clear:both;'></div>";
		}
	}

	//switch board for the page display
	if(isset($_GET['action']) && !isset($_GET['confirm']))
	{
		echo "<div id='main'>";
		switch($_GET['action'])
		{
			//table actions
			/////////////////////////////////////////////// create table
			case "table_create":
				$query = "SELECT name FROM sqlite_master WHERE type='table' AND name='".$_POST['tablename']."'";
				$results = $db->selectArray($query);
				if(sizeof($results)>0)
					$exists = true;
				else
					$exists = false;
				echo "<h2>Creating new table: '".$_POST['tablename']."'</h2>";
				if($_POST['tablefields']=="" || intval($_POST['tablefields'])<=0)
					echo "You must specify the number of table fields.";
				else if($_POST['tablename']=="")
					echo "You must specify a table name.";
				else if($exists)
					echo "Table of the same name already exists.";
				else
				{
					$num = intval($_POST['tablefields']);
					$name = $_POST['tablename'];
					echo "<form action='".PAGE."?action=table_create&confirm=1' method='post'>";
					echo "<input type='hidden' name='tablename' value='".$name."'/>";
					echo "<input type='hidden' name='rows' value='".$num."'/>";
					echo "<table border='0' cellpadding='2' cellspacing='1' class='viewTable'>";
					echo "<tr>";
					$headings = array("Field", "Type", "Primary Key", "Autoincrement", "Not NULL", "Default Value");
      			for($k=0; $k<count($headings); $k++)
						echo "<td class='tdheader'>" . $headings[$k] . "</td>";
					echo "</tr>";

					for($i=0; $i<$num; $i++)
					{
						$tdWithClass = "<td class='td" . ($i%2 ? "1" : "2") . "'>";
						echo "<tr>";
						echo $tdWithClass;
						echo "<input type='text' name='".$i."_field' style='width:200px;'/>";
						echo "</td>";
						echo $tdWithClass;
						echo "<select name='".$i."_type' id='".$i."_type' onchange='toggleAutoincrement(".$i.");'>";
						$types = unserialize(DATATYPES);
						for($z=0; $z<sizeof($types); $z++)
							echo "<option value='".$types[$z]."'>".$types[$z]."</option>";
						echo "</select>";
						echo "</td>";
						echo $tdWithClass;
						echo "<input type='checkbox' name='".$i."_primarykey' id='".$i."_primarykey' onclick='toggleNull(".$i.");'/> Yes";
						echo "</td>";
						echo $tdWithClass;
						echo "<input type='checkbox' name='".$i."_autoincrement' id='".$i."_autoincrement'/> Yes";
						echo "</td>";
						echo $tdWithClass;
						echo "<input type='checkbox' name='".$i."_notnull' id='".$i."_notnull'/> Yes";
						echo "</td>";
						echo $tdWithClass;
						echo "<input type='text' name='".$i."_defaultvalue' style='width:100px;'/>";
						echo "</td>";
						echo "</tr>";
					}
					echo "<tr>";
					echo "<td class='tdheader' style='text-align:right;' colspan='6'>";
					echo "<input type='submit' value='Create' class='btn'/> ";
					echo "<a href='".PAGE."'>Cancel</a>";
					echo "</td>";
					echo "</tr>";
					echo "</table>";
					echo "</form>";
				}
				break;
			/////////////////////////////////////////////// perform SQL query on table
			case "table_sql":
				$isSelect = false;
				if(isset($_POST['query']) && $_POST['query']!="")
				{
					$delimiter = $_POST['delimiter'];
					$queryStr = stripslashes($_POST['queryval']);
					$query = explode_sql($delimiter, $queryStr); //explode the query string into individual queries based on the delimiter

					for($i=0; $i<sizeof($query); $i++) //iterate through the queries exploded by the delimiter
					{
						if(str_replace(" ", "", str_replace("\n", "", str_replace("\r", "", $query[$i])))!="") //make sure this query is not an empty string
						{
							$startTime = microtime(true);
							if(strpos(strtolower($query[$i]), "select ")!==false)
							{
								$isSelect = true;
								$result = $db->selectArray($query[$i], "assoc");
							}
							else
							{
								$isSelect = false;
								$result = $db->query($query[$i]);
							}
							$endTime = microtime(true);
							$time = round(($endTime - $startTime), 4);

							echo "<div class='confirm'>";
							echo "<b>";
							if($result!==false)
							{
								if($isSelect)
								{
									$affected = sizeof($result);
									echo "Showing ".$affected." row(s). ";
								}
								else
								{
									$affected = $db->getAffectedRows();
									echo $affected." row(s) affected. ";
								}
								echo "(Query took ".$time." sec)</b><br/>";
							}
							else
							{
								echo "There is a problem with the syntax of your query ";
								echo "(Query was not executed)</b><br/>";
							}
							echo "<span style='font-size:11px;'>".$query[$i]."</span>";
							echo "</div><br/>";
							if($isSelect)
							{
								if(sizeof($result)>0)
								{
									$headers = array_keys($result[0]);

									echo "<table border='0' cellpadding='2' cellspacing='1' class='viewTable'>";
									echo "<tr>";
									for($j=0; $j<sizeof($headers); $j++)
									{
										echo "<td class='tdheader'>";
										echo $headers[$j];
										echo "</td>";
									}
									echo "</tr>";
									for($j=0; $j<sizeof($result); $j++)
									{
										$tdWithClass = "<td class='td".($j%2 ? "1" : "2")."'>";
										echo "<tr>";
										for($z=0; $z<sizeof($headers); $z++)
										{
											echo $tdWithClass;
											echo $result[$j][$headers[$z]];
											echo "</td>";
										}
										echo "</tr>";
									}
									echo "</table><br/><br/>";
								}
							}
						}
					}
				}
				else
				{
					$delimiter = ";";
					$queryStr = "SELECT * FROM ".$_GET['table']." WHERE 1";
				}

				echo "<fieldset>";
				echo "<legend><b>Run SQL query/queries on database '".$db->getName()."'</b></legend>";
				if(!isset($_GET['view']))
					echo "<form action='".PAGE."?table=".$_GET['table']."&action=table_sql' method='post'>";
				else
					echo "<form action='".PAGE."?table=".$_GET['table']."&action=table_sql&view=1' method='post'>";
				echo "<div style='float:left; width:70%;'>";
				echo "<textarea style='width:97%; height:300px;' name='queryval' id='queryval'>".$queryStr."</textarea>";
				echo "</div>";
				echo "<div style='float:left; width:28%; padding-left:10px;'>";
				echo "Fields<br/>";
				echo "<select multiple='multiple' style='width:100%;' id='fieldcontainer'>";
				$query = "PRAGMA table_info('".$_GET['table']."')";
				$result = $db->selectArray($query);
				for($i=0; $i<sizeof($result); $i++)
				{
					echo "<option value='".$result[$i][1]."'>".$result[$i][1]."</option>";
				}
				echo "</select>";
				echo "<input type='button' value='<<' onclick='moveFields();' class='btn'/>";
				echo "</div>";
				echo "<div style='clear:both;'></div>";
				echo "Delimiter <input type='text' name='delimiter' value='".$delimiter."' style='width:50px;'/> ";
				echo "<input type='submit' name='query' value='Go' class='btn'/>";
				echo "</form>";
				break;
			/////////////////////////////////////////////// empty table
			case "table_empty":
				echo "<form action='".PAGE."?action=table_empty&confirm=1' method='post'>";
				echo "<input type='hidden' name='tablename' value='".$_GET['table']."'/>";
				echo "<div class='confirm'>";
				echo "Are you sure you want to empty the table '".$_GET['table']."'?<br/><br/>";
				echo "<input type='submit' value='Confirm' class='btn'/> ";
				echo "<a href='".PAGE."'>Cancel</a>";
				echo "</div>";
				break;
			/////////////////////////////////////////////// drop table
			case "table_drop":
				echo "<form action='".PAGE."?action=table_drop&confirm=1' method='post'>";
				echo "<input type='hidden' name='tablename' value='".$_GET['table']."'/>";
				echo "<div class='confirm'>";
				echo "Are you sure you want to drop the table '".$_GET['table']."'?<br/><br/>";
				echo "<input type='submit' value='Confirm' class='btn'/> ";
				echo "<a href='".PAGE."'>Cancel</a>";
				echo "</div>";
				break;
			/////////////////////////////////////////////// drop view
			case "view_drop":
				echo "<form action='".PAGE."?action=view_drop&confirm=1' method='post'>";
				echo "<input type='hidden' name='viewname' value='".$_GET['table']."'/>";
				echo "<div class='confirm'>";
				echo "Are you sure you want to drop the view '".$_GET['table']."'?<br/><br/>";
				echo "<input type='submit' value='Confirm' class='btn'/> ";
				echo "<a href='".PAGE."'>Cancel</a>";
				echo "</div>";
				break;
			/////////////////////////////////////////////// export table
			case "table_export":
				echo "<form method='post' action='".PAGE."'>";
				echo "<fieldset style='float:left; width:260px; margin-right:20px;'><legend><b>Export</b></legend>";
				echo "<input type='hidden' value='".$_GET['table']."' name='single_table'/>";
				echo "<input type='radio' name='export_type' checked='checked' value='sql' onclick='toggleExports(\"sql\");'/> SQL";
				echo "<br/><input type='radio' name='export_type' value='csv' onclick='toggleExports(\"csv\");'/> CSV";
				echo "</fieldset>";
				
				echo "<fieldset style='float:left; max-width:350px;' id='exportoptions_sql'><legend><b>Options</b></legend>";
				echo "<input type='checkbox' checked='checked' name='structure'/> Export with structure ".helpLink("Export Structure to SQL File")."<br/>";
				echo "<input type='checkbox' checked='checked' name='data'/> Export with data ".helpLink("Export Data to SQL File")."<br/>";
				echo "<input type='checkbox' name='drop'/> Add DROP TABLE ".helpLink("Add Drop Table to Exported SQL File")."<br/>";
				echo "<input type='checkbox' checked='checked' name='transaction'/> Add TRANSACTION ".helpLink("Add Transaction to Exported SQL File")."<br/>";
				echo "<input type='checkbox' checked='checked' name='comments'/> Comments ".helpLink("Add Comments to Exported SQL File")."<br/>";
				echo "</fieldset>";
				
				echo "<fieldset style='float:left; max-width:350px; display:none;' id='exportoptions_csv'><legend><b>Options</b></legend>";
				echo "<div style='float:left;'>Fields terminated by</div>";
				echo "<input type='text' value=';' name='export_csv_fieldsterminated' style='float:right;'/>";
				echo "<div style='clear:both;'>";
				echo "<div style='float:left;'>Fields enclosed by</div>";
				echo "<input type='text' value='\"' name='export_csv_fieldsenclosed' style='float:right;'/>";
				echo "<div style='clear:both;'>";
				echo "<div style='float:left;'>Fields escaped by</div>";
				echo "<input type='text' value='\' name='export_csv_fieldsescaped' style='float:right;'/>";
				echo "<div style='clear:both;'>";
				echo "<div style='float:left;'>Replace NULL by</div>";
				echo "<input type='text' value='NULL' name='export_csv_replacenull' style='float:right;'/>";
				echo "<div style='clear:both;'>";
				echo "<input type='checkbox' name='export_csv_crlf'/> Remove CRLF characters within fields<br/>";
				echo "<input type='checkbox' checked='checked' name='export_csv_fieldnames'/> Put field names in first row";
				echo "</fieldset>";
				
				echo "<div style='clear:both;'></div>";
				echo "<br/><br/>";
				echo "<fieldset style='float:left;'><legend><b>Save As</b></legend>";
				echo "<input type='hidden' name='database_num' value='".$_SESSION[COOKIENAME.'currentDB']."'/>";
				$file = pathinfo($db->getPath());
				$name = $file['filename'];
				echo "<input type='text' name='filename' value='".$name.".".$_GET['table'].".".date("n-j-y").".dump' style='width:400px;'/> <input type='submit' name='export' value='Export' class='btn'/>";
				echo "</fieldset>";
				echo "</form>";
				break;
			/////////////////////////////////////////////// import table
			case "table_import":
				if(isset($_POST['import']))
				{
					echo "<div class='confirm'>";
					if($importSuccess===true)
						echo "Import was successful.";
					else
						echo $importSuccess;
					echo "</div><br/>";
				}
				echo "<form method='post' action='".PAGE."?table=".$_GET['table']."&action=table_import' enctype='multipart/form-data'>";
				echo "<fieldset style='float:left; width:260px; margin-right:20px;'><legend><b>Import into ".$_GET['table']."</b></legend>";
				echo "<input type='radio' name='import_type' checked='checked' value='sql' onclick='toggleImports(\"sql\");'/> SQL";
				echo "<br/><input type='radio' name='import_type' value='csv' onclick='toggleImports(\"csv\");'/> CSV";
				echo "</fieldset>";
				
				echo "<fieldset style='float:left; max-width:350px;' id='importoptions_sql'><legend><b>Options</b></legend>";
				echo "No options";
				echo "</fieldset>";
				
				echo "<fieldset style='float:left; max-width:350px; display:none;' id='importoptions_csv'><legend><b>Options</b></legend>";
				echo "<input type='hidden' value='".$_GET['table']."' name='single_table'/>";
				echo "<div style='float:left;'>Fields terminated by</div>";
				echo "<input type='text' value=';' name='import_csv_fieldsterminated' style='float:right;'/>";
				echo "<div style='clear:both;'>";
				echo "<div style='float:left;'>Fields enclosed by</div>";
				echo "<input type='text' value='\"' name='import_csv_fieldsenclosed' style='float:right;'/>";
				echo "<div style='clear:both;'>";
				echo "<div style='float:left;'>Fields escaped by</div>";
				echo "<input type='text' value='\' name='import_csv_fieldsescaped' style='float:right;'/>";
				echo "<div style='clear:both;'>";
				echo "<div style='float:left;'>NULL represented by</div>";
				echo "<input type='text' value='NULL' name='import_csv_replacenull' style='float:right;'/>";
				echo "<div style='clear:both;'>";
				echo "<input type='checkbox' checked='checked' name='import_csv_fieldnames'/> Field names in first row";
				echo "</fieldset>";
				
				echo "<div style='clear:both;'></div>";
				echo "<br/><br/>";
				
				echo "<fieldset><legend><b>File to import</b></legend>";
				echo "<input type='file' value='Choose File' name='file' style='background-color:transparent; border-style:none;'/> <input type='submit' value='Import' name='import' class='btn'/>";
				echo "</fieldset>";
				break;
			/////////////////////////////////////////////// rename table
			case "table_rename":
				echo "<form action='".PAGE."?action=table_rename&confirm=1' method='post'>";
				echo "<input type='hidden' name='oldname' value='".$_GET['table']."'/>";
				echo "Rename table '".$_GET['table']."' to <input type='text' name='newname' style='width:200px;'/> <input type='submit' value='Rename' name='rename' class='btn'/>";
				echo "</form>";
				break;
			/////////////////////////////////////////////// search table
			case "table_search":
				if(isset($_GET['done']))
				{
					$query = "PRAGMA table_info('".$_GET['table']."')";
					$result = $db->selectArray($query);
					$str = "";
					$j = 0;
					$arr = array();
					for($i=0; $i<sizeof($result); $i++)
					{
						$field = $result[$i][1];
						$operator = $_POST[$field.":operator"];
						$value = $_POST[$field];
						if($value!="" || $operator=="!= ''" || $operator=="= ''")
						{
							if($operator=="= ''" || $operator=="!= ''")
								$arr[$j] .= $field." ".$operator;
							else
								$arr[$j] .= $field." ".$operator." ".$db->quote($value);
							$j++;
						}
					}
					$query = "SELECT * FROM ".$_GET['table'];
					if(sizeof($arr)>0)
					{
						$query .= " WHERE ".$arr[0];
						for($i=1; $i<sizeof($arr); $i++)
						{
							$query .= " AND ".$arr[$i];
						}
					}
					$startTime = microtime(true);
					$result = $db->selectArray($query, "assoc");
					$endTime = microtime(true);
					$time = round(($endTime - $startTime), 4);

					echo "<div class='confirm'>";
					echo "<b>";
					if($result!==false)
					{
						$affected = sizeof($result);
						echo "Showing ".$affected." row(s). ";
						echo "(Query took ".$time." sec)</b><br/>";
					}
					else
					{
						echo "There is a problem with the syntax of your query ";
						echo "(Query was not executed)</b><br/>";
					}
					echo "<span style='font-size:11px;'>".$query."</span>";
					echo "</div><br/>";

					if(sizeof($result)>0)
					{
						$headers = array_keys($result[0]);

						echo "<table border='0' cellpadding='2' cellspacing='1' class='viewTable'>";
						echo "<tr>";
						for($j=0; $j<sizeof($headers); $j++)
						{
							echo "<td class='tdheader'>";
							echo $headers[$j];
							echo "</td>";
						}
						echo "</tr>";
						for($j=0; $j<sizeof($result); $j++)
						{
							$tdWithClass = "<td class='td".($j%2 ? "1" : "2")."'>";
							echo "<tr>";
							for($z=0; $z<sizeof($headers); $z++)
							{
								echo $tdWithClass;
								echo $result[$j][$headers[$z]];
								echo "</td>";
							}
							echo "</tr>";
						}
						echo "</table><br/><br/>";
					}
					
					if(!isset($_GET['view']))
						echo "<a href='".PAGE."?table=".$_GET['table']."&action=table_search'>Do Another Search</a>";
					else
						echo "<a href='".PAGE."?table=".$_GET['table']."&action=table_search&view=1'>Do Another Search</a>";
				}
				else
				{
					$query = "PRAGMA table_info('".$_GET['table']."')";
					$result = $db->selectArray($query);
					
					if(!isset($_GET['view']))
						echo "<form action='".PAGE."?table=".$_GET['table']."&action=table_search&done=1' method='post'>";
					else
						echo "<form action='".PAGE."?table=".$_GET['table']."&action=table_search&view=1&done=1' method='post'>";
						
					echo "<table border='0' cellpadding='2' cellspacing='1' class='viewTable'>";
					echo "<tr>";
					echo "<td class='tdheader'>Field</td>";
					echo "<td class='tdheader'>Type</td>";
					echo "<td class='tdheader'>Operator</td>";
					echo "<td class='tdheader'>Value</td>";
					echo "</tr>";

					for($i=0; $i<sizeof($result); $i++)
					{
					  $field = $result[$i][1];
					  $type = $result[$i][2];
					  $tdWithClass = "<td class='td".($i%2 ? "1" : "2")."'>";
					  $tdWithClassLeft = "<td class='td".($i%2 ? "1" : "2")."' style='text-align:left;'>";
					  echo "<tr>";
					  echo $tdWithClassLeft;
					  echo $field;
					  echo "</td>";
					  echo $tdWithClassLeft;
					  echo $type;
					  echo "</td>";
					  echo $tdWithClassLeft;
					  echo "<select name='".$field.":operator'>";
					  echo "<option value='='>=</option>";
					  if($type=="INTEGER" || $type=="REAL")
					  {
						  echo "<option value='>'>></option>";
						  echo "<option value='>='>>=</option>";
						  echo "<option value='<'><</option>";
						  echo "<option value='<='><=</option>";
					  }
					  else if($type=="TEXT" || $type=="BLOB")
					  {
						  echo "<option value='= '''>= ''</option>";
						  echo "<option value='!= '''>!= ''</option>";
					  }
					  echo "<option value='!='>!=</option>";
					  if($type=="TEXT" || $type=="BLOB")
						  echo "<option value='LIKE' selected='selected'>LIKE</option>";
					  else
						  echo "<option value='LIKE'>LIKE</option>";
					  echo "<option value='NOT LIKE'>NOT LIKE</option>";
					  echo "</select>";
					  echo "</td>";
					  echo $tdWithClassLeft;
					  if($type=="INTEGER" || $type=="REAL" || $type=="NULL")
						  echo "<input type='text' name='".$field."'/>";
					  else
						  echo "<textarea name='".$field."' wrap='hard' rows='1' cols='60'></textarea>";
					  echo "</td>";
					  echo "</tr>";
					}
					echo "<tr>";
					echo "<td class='tdheader' style='text-align:right;' colspan='4'>";
					echo "<input type='submit' value='Search' class='btn'/>";
					echo "</td>";
					echo "</tr>";
					echo "</table>";
					echo "</form>";
				}
				break;
			//row actions
			/////////////////////////////////////////////// view row
			case "row_view":
				if(!isset($_POST['startRow']))
					$_POST['startRow'] = 0;

				if(isset($_POST['numRows']))
					$_SESSION[COOKIENAME.'numRows'] = $_POST['numRows'];

				if(!isset($_SESSION[COOKIENAME.'numRows']))
					$_SESSION[COOKIENAME.'numRows'] = 30;
				
				if(isset($_SESSION[COOKIENAME.'currentTable']) && $_SESSION[COOKIENAME.'currentTable']!=$_GET['table'])
				{
					unset($_SESSION[COOKIENAME.'sort']);
					unset($_SESSION[COOKIENAME.'order']);	
				}
				if(isset($_POST['viewtype']))
				{
					$_SESSION[COOKIENAME.'viewtype'] = $_POST['viewtype'];	
				}
				
				$query = "SELECT Count(*) FROM ".$_GET['table'];
				$rowCount = $db->select($query);
				$rowCount = intval($rowCount[0]);
				$lastPage = intval($rowCount / $_SESSION[COOKIENAME.'numRows']);
				$remainder = intval($rowCount % $_SESSION[COOKIENAME.'numRows']);
				if($remainder==0)
					$remainder = $_SESSION[COOKIENAME.'numRows'];
				
				echo "<div style='overflow:hidden;'>";
				//previous button
				if($_POST['startRow']>0)
				{
					echo "<div style='float:left; overflow:hidden;'>";
					echo "<form action='".PAGE."?action=row_view&table=".$_GET['table']."' method='post'>";
					echo "<input type='hidden' name='startRow' value='0'/>";
					echo "<input type='hidden' name='numRows' value='".$_SESSION[COOKIENAME.'numRows']."'/> ";
					echo "<input type='submit' value='&larr;&larr;' name='previous' class='btn'/> ";
					echo "</form>";
					echo "</div>";
					echo "<div style='float:left; overflow:hidden; margin-right:20px;'>";
					echo "<form action='".PAGE."?action=row_view&table=".$_GET['table']."' method='post'>";
					echo "<input type='hidden' name='startRow' value='".intval($_POST['startRow']-$_SESSION[COOKIENAME.'numRows'])."'/>";
					echo "<input type='hidden' name='numRows' value='".$_SESSION[COOKIENAME.'numRows']."'/> ";
					echo "<input type='submit' value='&larr;' name='previous_full' class='btn'/> ";
					echo "</form>";
					echo "</div>";
				}
				
				//show certain number buttons
				echo "<div style='float:left; overflow:hidden;'>";
				echo "<form action='".PAGE."?action=row_view&table=".$_GET['table']."' method='post'>";
				echo "<input type='submit' value='Show : ' name='show' class='btn'/> ";
				echo "<input type='text' name='numRows' style='width:50px;' value='".$_SESSION[COOKIENAME.'numRows']."'/> ";
				echo "row(s) starting from record # ";
				if(intval($_POST['startRow']+$_SESSION[COOKIENAME.'numRows']) < $rowCount)
					echo "<input type='text' name='startRow' style='width:90px;' value='".intval($_POST['startRow']+$_SESSION[COOKIENAME.'numRows'])."'/>";
				else
					echo "<input type='text' name='startRow' style='width:90px;' value='0'/>";
				echo " as a ";
				echo "<select name='viewtype'>";
				if(!isset($_SESSION[COOKIENAME.'viewtype']) || $_SESSION[COOKIENAME.'viewtype']=="table")
				{
					echo "<option value='table' selected='selected'>Table</option>";
					echo "<option value='chart'>Chart</option>";
				}
				else
				{
					echo "<option value='table'>Table</option>";
					echo "<option value='chart' selected='selected'>Chart</option>";
				}
				echo "</select>";
				echo "</form>";
				echo "</div>";
				
				//next button
				if(intval($_POST['startRow']+$_SESSION[COOKIENAME.'numRows'])<$rowCount)
				{
					echo "<div style='float:left; overflow:hidden; margin-left:20px; '>";
					echo "<form action='".PAGE."?action=row_view&table=".$_GET['table']."' method='post'>";
					echo "<input type='hidden' name='startRow' value='".intval($_POST['startRow']+$_SESSION[COOKIENAME.'numRows'])."'/>";
					echo "<input type='hidden' name='numRows' value='".$_SESSION[COOKIENAME.'numRows']."'/> ";
					echo "<input type='submit' value='&rarr;' name='next' class='btn'/> ";
					echo "</form>";
					echo "</div>";
					echo "<div style='float:left; overflow:hidden;'>";
					echo "<form action='".PAGE."?action=row_view&table=".$_GET['table']."' method='post'>";
					echo "<input type='hidden' name='startRow' value='".intval($rowCount-$remainder)."'/>";
					echo "<input type='hidden' name='numRows' value='".$_SESSION[COOKIENAME.'numRows']."'/> ";
					echo "<input type='submit' value='&rarr;&rarr;' name='next_full' class='btn'/> ";
					echo "</form>";
					echo "</div>";
				}
				echo "<div style='clear:both;'></div>";
				echo "</div>";
				
				if(!isset($_GET['sort']))
					$_GET['sort'] = NULL;
				if(!isset($_GET['order']))
					$_GET['order'] = NULL;

				$table = $_GET['table'];
				$numRows = $_SESSION[COOKIENAME.'numRows'];
				$startRow = $_POST['startRow'];
				if(isset($_GET['sort']))
				{
					$_SESSION[COOKIENAME.'sort'] = $_GET['sort'];
					$_SESSION[COOKIENAME.'currentTable'] = $_GET['table'];
				}
				if(isset($_GET['order']))
				{
					$_SESSION[COOKIENAME.'order'] = $_GET['order'];
					$_SESSION[COOKIENAME.'currentTable'] = $_GET['table'];
				}
				$_SESSION[COOKIENAME.'numRows'] = $numRows;
				$query = "SELECT *, ROWID FROM ".$table;
				$queryDisp = "SELECT * FROM ".$table;
				$queryAdd = "";
				if(isset($_SESSION[COOKIENAME.'sort']))
					$queryAdd .= " ORDER BY ".$_SESSION[COOKIENAME.'sort'];
				if(isset($_SESSION[COOKIENAME.'order']))
					$queryAdd .= " ".$_SESSION[COOKIENAME.'order'];
				$queryAdd .= " LIMIT ".$startRow.", ".$numRows;
				$query .= $queryAdd;
				$queryDisp .= $queryAdd;
				$startTime = microtime(true);
				$arr = $db->selectArray($query);
				$endTime = microtime(true);
				$time = round(($endTime - $startTime), 4);
				$total = $db->numRows($table);

				if(sizeof($arr)>0)
				{
					echo "<br/><div class='confirm'>";
					echo "<b>Showing rows ".$startRow." - ".($startRow + sizeof($arr)-1)." (".$total." total, Query took ".$time." sec)</b><br/>";
					echo "<span style='font-size:11px;'>".$queryDisp."</span>";
					echo "</div><br/>";
					
					if(isset($_GET['view']))
					{
						echo "'".$_GET['table']."' is a view, which means it is a SELECT statement treated as a read-only table. You may not edit or insert records. <a href='http://en.wikipedia.org/wiki/View_(database)' target='_blank'>http://en.wikipedia.org/wiki/View_(database)</a>"; 
						echo "<br/><br/>";	
					}
					
					$query = "PRAGMA table_info('".$table."')";
					$result = $db->selectArray($query);
					$rowidColumn = sizeof($result);
					
					if(!isset($_SESSION[COOKIENAME.'viewtype']) || $_SESSION[COOKIENAME.'viewtype']=="table")
					{
						echo "<form action='".PAGE."?action=row_editordelete&table=".$table."' method='post' name='checkForm'>";
						echo "<table border='0' cellpadding='2' cellspacing='1' class='viewTable'>";
						echo "<tr>";
						if(!isset($_GET['view']))
							echo "<td colspan='3'></td>";
	
						for($i=0; $i<sizeof($result); $i++)
						{
							echo "<td class='tdheader'>";
							if(!isset($_GET['view']))
								echo "<a href='".PAGE."?action=row_view&table=".$table."&sort=".$result[$i][1];
							else
								echo "<a href='".PAGE."?action=row_view&table=".$table."&view=1&sort=".$result[$i][1];
							if(isset($_SESSION[COOKIENAME.'sort']))
								$orderTag = ($_SESSION[COOKIENAME.'sort']==$result[$i][1] && $_SESSION[COOKIENAME.'order']=="ASC") ? "DESC" : "ASC";
							else
								$orderTag = "ASC";
							echo "&order=".$orderTag;
							echo "'>".$result[$i][1]."</a>";
							if(isset($_SESSION[COOKIENAME.'sort']) && $_SESSION[COOKIENAME.'sort']==$result[$i][1])
								echo (($_SESSION[COOKIENAME.'order']=="ASC") ? " <b>&uarr;</b>" : " <b>&darr;</b>");
							echo "</td>";
						}
						echo "</tr>";
	
						for($i=0; $i<sizeof($arr); $i++)
						{
							// -g-> $pk will always be the last column in each row of the array because we are doing a "SELECT *, ROWID FROM ..."
							$pk = $arr[$i][$rowidColumn];
							$tdWithClass = "<td class='td".($i%2 ? "1" : "2")."'>";
							$tdWithClassLeft = "<td class='td".($i%2 ? "1" : "2")."' style='text-align:left;'>";
							echo "<tr>";
							if(!isset($_GET['view']))
							{
								echo $tdWithClass;
								echo "<input type='checkbox' name='check[]' value='".$pk."' id='check_".$i."'/>";
								echo "</td>";
								echo $tdWithClass;
								// -g-> Here, we need to put the ROWID in as the link for both the edit and delete.
								echo "<a href='".PAGE."?table=".$table."&action=row_editordelete&pk=".$pk."&type=edit'>edit</a>";
								echo "</td>";
								echo $tdWithClass;
								echo "<a href='".PAGE."?table=".$table."&action=row_editordelete&pk=".$pk."&type=delete' style='color:red;'>delete</a>";
								echo "</td>";
							}
							for($j=0; $j<sizeof($result); $j++)
							{
								if(strtolower($result[$j][2])=="integer" || strtolower($result[$j][2])=="float" || strtolower($result[$j][2])=="real")
									echo $tdWithClass;
								else
									echo $tdWithClassLeft;
								// -g-> although the inputs do not interpret HTML on the way "in", when we print the contents of the database the interpretation cannot be avoided.
								// di - i don't understand how SQLite returns null values. I played around with the conditional here and couldn't get empty strings to differeniate from actual null values...
								if($arr[$i][$j]==NULL)
									echo "<i>NULL</i>";
								else
									echo $db->formatString($arr[$i][$j]);
								echo "</td>";
							}
							echo "</tr>";
						}
						echo "</table>";
						if(!isset($_GET['view']))
						{
							echo "<a onclick='checkAll()'>Check All</a> / <a onclick='uncheckAll()'>Uncheck All</a> <i>With selected:</i> ";
							echo "<select name='type'>";
							echo "<option value='edit'>Edit</option>";
							echo "<option value='delete'>Delete</option>";
							echo "</select> ";
							echo "<input type='submit' value='Go' name='massGo' class='btn'/>";
						}
						echo "</form>";
					}
					else
					{
						if(!isset($_SESSION[COOKIENAME.$_GET['table'].'chartlabels']))
						{
							for($i=0; $i<sizeof($result); $i++)
							{
								if(strtolower($result[$i][2])=="text")
									$_SESSION[COOKIENAME.$_GET['table'].'chartlabels'] = $i;
							}
						}
						if(!isset($_SESSION[COOKIENAME.'chartlabels']))
							$_SESSION[COOKIENAME.'chartlabels'] = 0;
							
						if(!isset($_SESSION[COOKIENAME.$_GET['table'].'chartvalues']))
						{
							for($i=0; $i<sizeof($result); $i++)
							{
								if(strtolower($result[$i][2])=="integer" || strtolower($result[$i][2])=="float" || strtolower($result[$i][2])=="real")
									$_SESSION[COOKIENAME.$_GET['table'].'chartvalues'] = $i;
							}
						}
						
						if(!isset($_SESSION[COOKIENAME.'charttype']))
							$_SESSION[COOKIENAME.'charttype'] = "bar";
							
						if(isset($_POST['chartsettings']))
						{
							$_SESSION[COOKIENAME.'charttype'] = $_POST['charttype'];	
							$_SESSION[COOKIENAME.$_GET['table'].'chartlabels'] = $_POST['chartlabels'];
							$_SESSION[COOKIENAME.$_GET['table'].'chartvalues'] = $_POST['chartvalues'];
						}
						//begin chart view
						?>
						<script type='text/javascript' src='https://www.google.com/jsapi'></script>
						<script type='text/javascript'>
						google.load('visualization', '1.0', {'packages':['corechart']});
						google.setOnLoadCallback(drawChart);
						function drawChart()
						{
							var data = new google.visualization.DataTable();
							data.addColumn('string', '<?php echo $result[$_SESSION[COOKIENAME.$_GET['table'].'chartlabels']][1]; ?>');
							data.addColumn('number', '<?php echo $result[$_SESSION[COOKIENAME.$_GET['table'].'chartvalues']][1]; ?>');
							data.addRows([
							<?php
							for($i=0; $i<sizeof($arr); $i++)
							{
								$label = str_replace("'", "", $db->formatString($arr[$i][$_SESSION[COOKIENAME.$_GET['table'].'chartlabels']]));
								$value = $db->formatString($arr[$i][$_SESSION[COOKIENAME.$_GET['table'].'chartvalues']]);
								
								if($value==NULL || $value=="")
									$value = 0;
									
								echo "['".$label."', ".$value."]";
								if($i<sizeof($arr)-1)
									echo ",";
							}
							$height = (sizeof($arr)+1) * 30;
							if($height>1000)
								$height = 1000;
							else if($height<300)
								$height = 300;
							if($_SESSION[COOKIENAME.'charttype']=="pie")
								$height = 800;
							?>
							]);
							var chartWidth = document.getElementById("content").offsetWidth - document.getElementById("chartsettingsbox").offsetWidth - 100;
							if(chartWidth>1000)
								chartWidth = 1000;
								
							var options = 
							{
								'width':chartWidth,
								'height':<?php echo $height; ?>,
								'title':'<?php echo $result[$_SESSION[COOKIENAME.$_GET['table'].'chartlabels']][1]." vs ".$result[$_SESSION[COOKIENAME.$_GET['table'].'chartvalues']][1]; ?>'
							};
							<?php
							if($_SESSION[COOKIENAME.'charttype']=="bar")
								echo "var chart = new google.visualization.BarChart(document.getElementById('chart_div'));";
							else if($_SESSION[COOKIENAME.'charttype']=="pie")
								echo "var chart = new google.visualization.PieChart(document.getElementById('chart_div'));";
							else
								echo "var chart = new google.visualization.LineChart(document.getElementById('chart_div'));";
							?>
							chart.draw(data, options);
						}
						</script>
						<div id="chart_div" style="float:left;">If you can read this, it means the chart could not be generated. The data you are trying to view may not be appropriate for a chart.</div>
						<?php
						echo "<fieldset style='float:right; text-align:center;' id='chartsettingsbox'><legend><b>Chart Settings</b></legend>";
						echo "<form action='".PAGE."?action=row_view&table=".$_GET['table']."' method='post'>";
						echo "Chart Type: <select name='charttype'>";
						echo "<option value='bar'";
						if($_SESSION[COOKIENAME.'charttype']=="bar")
							echo " selected='selected'";
						echo ">Bar Chart</option>";
						echo "<option value='pie'";
						if($_SESSION[COOKIENAME.'charttype']=="pie")
							echo " selected='selected'";
						echo ">Pie Chart</option>";
						echo "<option value='line'";
						if($_SESSION[COOKIENAME.'charttype']=="line")
							echo " selected='selected'";
						echo ">Line Chart</option>";
						echo "</select>";
						echo "<br/><br/>";
						echo "Labels: <select name='chartlabels'>";
						for($i=0; $i<sizeof($result); $i++)
						{
							if(isset($_SESSION[COOKIENAME.$_GET['table'].'chartlabels']) && $_SESSION[COOKIENAME.$_GET['table'].'chartlabels']==$i)
								echo "<option value='".$i."' selected='selected'>".$result[$i][1]."</option>";
							else
								echo "<option value='".$i."'>".$result[$i][1]."</option>";
						}
						echo "</select>";
						echo "<br/><br/>";
						echo "Values: <select name='chartvalues'>";
						for($i=0; $i<sizeof($result); $i++)
						{
							if(strtolower($result[$i][2])=="integer" || strtolower($result[$i][2])=="float" || strtolower($result[$i][2])=="real")
							{
								if(isset($_SESSION[COOKIENAME.$_GET['table'].'chartvalues']) && $_SESSION[COOKIENAME.$_GET['table'].'chartvalues']==$i)
									echo "<option value='".$i."' selected='selected'>".$result[$i][1]."</option>";
								else
									echo "<option value='".$i."'>".$result[$i][1]."</option>";
							}
						}
						echo "</select>";
						echo "<br/><br/>";
						echo "<input type='submit' name='chartsettings' value='Update' class='btn'/>";
						echo "</form>";
						echo "</fieldset>";
						echo "<div style='clear:both;'></div>";
						//end chart view
					}
				}
				else if($rowCount>0)//no rows - do nothing
				{
					echo "<br/><br/>There are no rows in the table for the range you selected.";
				}
				else
				{
					echo "<br/><br/>This table is empty. <a href='".PAGE."?table=".$_GET['table']."&action=row_create'>Click here</a> to insert rows.";
				}

				break;
			/////////////////////////////////////////////// create row
			case "row_create":
				$fieldStr = "";
				echo "<form action='".PAGE."?table=".$_GET['table']."&action=row_create' method='post'>";
				echo "Restart insertion with ";
				echo "<select name='num'>";
				for($i=1; $i<=40; $i++)
				{
					if(isset($_POST['num']) && $_POST['num']==$i)
						echo "<option value='".$i."' selected='selected'>".$i."</option>";
					else
						echo "<option value='".$i."'>".$i."</option>";
				}
				echo "</select>";
				echo " rows ";
				echo "<input type='submit' value='Go' class='btn'/>";
				echo "</form>";
				echo "<br/>";
				$query = "PRAGMA table_info('".$_GET['table']."')";
				$result = $db->selectArray($query);
				echo "<form action='".PAGE."?table=".$_GET['table']."&action=row_create&confirm=1' method='post'>";
				if(isset($_POST['num']))
					$num = $_POST['num'];
				else
					$num = 1;
				echo "<input type='hidden' name='numRows' value='".$num."'/>";
				for($j=0; $j<$num; $j++)
				{
					if($j>0)
						echo "<input type='checkbox' value='ignore' name='".$j.":ignore' id='".$j."_ignore' checked='checked'/> Ignore<br/>";
					echo "<table border='0' cellpadding='2' cellspacing='1' class='viewTable'>";
					echo "<tr>";
					echo "<td class='tdheader'>Field</td>";
					echo "<td class='tdheader'>Type</td>";
					echo "<td class='tdheader'>Function</td>";
					echo "<td class='tdheader'>Null</td>";
					echo "<td class='tdheader'>Value</td>";
					echo "</tr>";

					for($i=0; $i<sizeof($result); $i++)
					{
						$field = $result[$i][1];
						if($j==0)
							$fieldStr .= ":".$field;
						$type = strtolower($result[$i][2]);
						$scalarField = $type=="integer" || $type=="real" || $type=="null";
						$tdWithClass = "<td class='td".($i%2 ? "1" : "2")."'>";
						$tdWithClassLeft = "<td class='td".($i%2 ? "1" : "2")."' style='text-align:left;'>";
						echo "<tr>";
						echo $tdWithClassLeft;
						echo $field;
						echo "</td>";
						echo $tdWithClassLeft;
						echo $type;
						echo "</td>";
						echo $tdWithClassLeft;
						echo "<select name='function_".$j."_".$field."' onchange='notNull(\"".$j.":".$field."_null\");'>";
						echo "<option value=''></option>";
						$functions = array_merge(unserialize(FUNCTIONS), $db->getUserFunctions());
						for($z=0; $z<sizeof($functions); $z++)
						{
							echo "<option value='".$functions[$z]."'>".$functions[$z]."</option>";
						}
						echo "</select>";
						echo "</td>";
						//we need to have a column dedicated to nulls -di
						echo $tdWithClassLeft;
						if($result[$i][3]==0)
						{
							if($result[$i][4]==NULL)
								echo "<input type='checkbox' name='".$j.":".$field."_null' id='".$j.":".$field."_null' checked='checked' onclick='disableText(this, \"".$j.":".$field."\");'/>";
							else
								echo "<input type='checkbox' name='".$j.":".$field."_null' id='".$j.":".$field."_null' onclick='disableText(this, \"".$j.":".$field."\");'/>";
						}
						echo "</td>";
						echo $tdWithClassLeft;
						// 22 August 2011: gkf fixed bug #55. The form is now prepopulated with the default values
						//                 so that the insert proceeds normally.
						// 22 August 2011: gkf fixed bug #53. The form now displays more of the text.
						// 19 October 2011: di fixed the bug caused by the previous fix where the null column does not exist anymore
						$type = strtolower($type);
						if($scalarField)
							echo "<input type='text' id='".$j.":".$field."' name='".$j.":".$field."' value='".deQuoteSQL($result[$i][4])."' onblur='changeIgnore(this, \"".$j."_ignore\");' onclick='notNull(\"".$j.":".$field."_null\");'/>";
						else
							echo "<textarea id='".$j.":".$field."' name='".$j.":".$field."' rows='5' cols='60' onclick='notNull(\"".$j.":".$field."_null\");' onblur='changeIgnore(this, \"".$j."_ignore\");'>".deQuoteSQL($result[$i][4])."</textarea>";
            		echo "</td>";
            		echo "</tr>";
					}
					echo "<tr>";
					echo "<td class='tdheader' style='text-align:right;' colspan='5'>";
					echo "<input type='submit' value='Insert' class='btn'/>";
					echo "</td>";
					echo "</tr>";
					echo "</table><br/>";
				}
				$fieldStr = substr($fieldStr, 1);
				echo "<input type='hidden' name='fields' value='".$fieldStr."'/>";
				echo "</form>";
				break;
			/////////////////////////////////////////////// edit or delete row
			case "row_editordelete":
				if(isset($_POST['check']))
					$pks = $_POST['check'];
				else if(isset($_GET['pk']))
					$pks = array($_GET['pk']);
				$str = $pks[0];
				$pkVal = $pks[0];
				for($i=1; $i<sizeof($pks); $i++)
				{
					$str .= ", ".$pks[$i];
					$pkVal .= ":".$pks[$i];
				}
				if($str=="") //nothing was selected so show an error
				{
					echo "<div class='confirm'>";
					echo "Error: You did not select anything.";
					echo "</div>";
					echo "<br/><br/><a href='".PAGE."?table=".$_GET['table']."&action=row_view'>Return</a>";
				}
				else
				{
					if((isset($_POST['type']) && $_POST['type']=="edit") || (isset($_GET['type']) && $_GET['type']=="edit")) //edit
					{
						echo "<form action='".PAGE."?table=".$_GET['table']."&action=row_edit&confirm=1&pk=".$pkVal."' method='post'>";
						$query = "PRAGMA table_info('".$_GET['table']."')";
						$result = $db->selectArray($query);

						//build the POST array of fields
						$fieldStr = $result[0][1];
						for($j=1; $j<sizeof($result); $j++)
							$fieldStr .= ":".$result[$j][1];

						echo "<input type='hidden' name='fieldArray' value='".$fieldStr."'/>";

						for($j=0; $j<sizeof($pks); $j++)
						{
							$query = "SELECT * FROM ".$_GET['table']." WHERE ROWID = ".$pks[$j];
							$result1 = $db->select($query);

							echo "<table border='0' cellpadding='2' cellspacing='1' class='viewTable'>";
							echo "<tr>";
							echo "<td class='tdheader'>Field</td>";
							echo "<td class='tdheader'>Type</td>";
							echo "<td class='tdheader'>Function</td>";
							echo "<td class='tdheader'>Null</td>";
							echo "<td class='tdheader'>Value</td>";
							echo "</tr>";

							for($i=0; $i<sizeof($result); $i++)
							{
								$field = $result[$i][1];
								$type = $result[$i][2];
								$value = $result1[$i];
								$tdWithClass = "<td class='td".($i%2 ? "1" : "2")."'>";
								$tdWithClassLeft = "<td class='td".($i%2 ? "1" : "2")."' style='text-align:left;'>";
								echo "<tr>";
								echo $tdWithClass;
								echo $field;
								echo "</td>";
								echo $tdWithClass;
								echo $type;
								echo "</td>";
								echo $tdWithClassLeft;
								echo "<select name='function_".$pks[$j]."_".$field."' onchange='notNull(\"".$pks[$j].":".$field."_null\");'>";
								echo "<option value=''></option>";
								$functions = array_merge(unserialize(FUNCTIONS), $db->getUserFunctions());
								for($z=0; $z<sizeof($functions); $z++)
								{
									echo "<option value='".$functions[$z]."'>".$functions[$z]."</option>";
								}
								echo "</select>";
								echo "</td>";
								echo $tdWithClassLeft;
								if($result[$i][3]==0)
								{
									if($value==NULL)
										echo "<input type='checkbox' name='".$pks[$j].":".$field."_null' id='".$pks[$j].":".$field."_null' checked='checked'/>";
									else
										echo "<input type='checkbox' name='".$pks[$j].":".$field."_null' id='".$pks[$j].":".$field."_null'/>";
								}
								echo "</td>";
								echo $tdWithClassLeft;
								if($type=="INTEGER" || $type=="REAL" || $type=="NULL")
									echo "<input type='text' name='".$pks[$j].":".$field."' value='".$db->formatString($value)."' onblur='changeIgnore(this, \"".$j."\", \"".$pks[$j].":".$field."_null\")' />";
								else
									echo "<textarea name='".$pks[$j].":".$field."' wrap='hard' rows='1' cols='60' onblur='changeIgnore(this, \"".$j."\", \"".$pks[$j].":".$field."_null\")'>".$db->formatString($value)."</textarea>";
								echo "</td>";
								echo "</tr>";
							}
							echo "<tr>";
							echo "<td class='tdheader' style='text-align:right;' colspan='5'>";
							echo "<input type='submit' name='new_row' value='Insert As New Row' class='btn'/> ";
							echo "<input type='submit' value='Save Changes' class='btn'/> ";
							echo "<a href='".PAGE."?table=".$_GET['table']."&action=row_view'>Cancel</a>";
							echo "</td>";
							echo "</tr>";
							echo "</table>";
							echo "<br/>";
						}
						echo "</form>";
					}
					else //delete
					{
						echo "<form action='".PAGE."?table=".$_GET['table']."&action=row_delete&confirm=1&pk=".$pkVal."' method='post'>";
						echo "<div class='confirm'>";
						echo "Are you sure you want to delete row(s) ".$str." from table '".$_GET['table']."'?<br/><br/>";
						echo "<input type='submit' value='Confirm' class='btn'/> ";
						echo "<a href='".PAGE."?table=".$_GET['table']."&action=row_view'>Cancel</a>";
						echo "</div>";
					}
				}
				break;
			//column actions
			/////////////////////////////////////////////// view column
			case "column_view":
				$query = "PRAGMA table_info('".$_GET['table']."')";
				$result = $db->selectArray($query);

				echo "<form action='".PAGE."?table=".$_GET['table']."&action=column_delete' method='post' name='checkForm'>";
				echo "<table border='0' cellpadding='2' cellspacing='1' class='viewTable'>";
				echo "<tr>";
				if(!isset($_GET['view']))
					echo "<td colspan='3'></td>";
				echo "<td class='tdheader'>Column #</td>";
				echo "<td class='tdheader'>Field</td>";
				echo "<td class='tdheader'>Type</td>";
				echo "<td class='tdheader'>Not Null</td>";
				echo "<td class='tdheader'>Default Value</td>";
				echo "<td class='tdheader'>Primary Key</td>";
				echo "</tr>";

				for($i=0; $i<sizeof($result); $i++)
				{
					$colVal = $result[$i][0];
					$fieldVal = $result[$i][1];
					$typeVal = $result[$i][2];
					$notnullVal = $result[$i][3];
					$defaultVal = $result[$i][4];
					$primarykeyVal = $result[$i][5];

					if(intval($notnullVal)!=0)
						$notnullVal = "yes";
					else
						$notnullVal = "no";
					if(intval($primarykeyVal)!=0)
						$primarykeyVal = "yes";
					else
						$primarykeyVal = "no";

					$tdWithClass = "<td class='td".($i%2 ? "1" : "2")."'>";
					$tdWithClassLeft = "<td class='td".($i%2 ? "1" : "2")."' style='text-align:left;'>";
					echo "<tr>";
					if(!isset($_GET['view']))
					{
						echo $tdWithClass;
						echo "<input type='checkbox' name='check[]' value='".$fieldVal."' id='check_".$i."'/>";
						echo "</td>";
						echo $tdWithClass;
						echo "<a href='".PAGE."?table=".$_GET['table']."&action=column_edit&pk=".$fieldVal."'>edit</a>";
						echo "</td>";
						echo $tdWithClass;
						echo "<a href='".PAGE."?table=".$_GET['table']."&action=column_delete&pk=".$fieldVal."' style='color:red;'>delete</a>";
						echo "</td>";
					}
					echo $tdWithClass;
					echo $colVal;
					echo "</td>";
					echo $tdWithClassLeft;
					echo $fieldVal;
					echo "</td>";
					echo $tdWithClassLeft;
					echo $typeVal;
					echo "</td>";
					echo $tdWithClassLeft;
					echo $notnullVal;
					echo "</td>";
					echo $tdWithClassLeft;
					echo $defaultVal;
					echo "</td>";
					echo $tdWithClassLeft;
					echo $primarykeyVal;
					echo "</td>";
					echo "</tr>";
				}

				echo "</table>";
				if(!isset($_GET['view']))
				{
					echo "<a onclick='checkAll()'>Check All</a> / <a onclick='uncheckAll()'>Uncheck All</a> <i>With selected:</i> ";
					echo "<select name='massType'>";
					//echo "<option value='edit'>Edit</option>";
					echo "<option value='delete'>Delete</option>";
					echo "</select> ";
					echo "<input type='hidden' name='structureDel' value='true'/>";
					echo "<input type='submit' value='Go' name='massGo' class='btn'/>";
				}
				echo "</form>";
				if(!isset($_GET['view']))
				{
					echo "<br/>";
					echo "<form action='".PAGE."?table=".$_GET['table']."&action=column_create' method='post'>";
					echo "<input type='hidden' name='tablename' value='".$_GET['table']."'/>";
					echo "Add <input type='text' name='tablefields' style='width:30px;' value='1'/> field(s) at end of table <input type='submit' value='Go' name='addfields' class='btn'/>";
					echo "</form>";
				}
				
				$query = "SELECT sql FROM sqlite_master WHERE name='".$_GET['table']."'";
				$master = $db->selectArray($query);
				
				echo "<br/>";
				if(!isset($_GET['view']))
					$typ = "table";
				else
					$typ = "view";
				echo "<br/>";
				echo "<div class='confirm'>";
				echo "<b>Query used to create this ".$typ."</b><br/>";
				echo "<span style='font-size:11px;'>".$master[0]['sql']."</span>";
				echo "</div>";
				echo "<br/>";
				if(!isset($_GET['view']))
				{
					echo "<br/><hr/><br/>";
					//$query = "SELECT * FROM sqlite_master WHERE type='index' AND tbl_name='".$_GET['table']."'";
					$query = "PRAGMA index_list(".$_GET['table'].")";
					$result = $db->selectArray($query);
					if(sizeof($result)>0)
					{
						echo "<h2>Indexes:</h2>";
						echo "<table border='0' cellpadding='2' cellspacing='1' class='viewTable'>";
						echo "<tr>";
						echo "<td colspan='1'>";
						echo "</td>";
						echo "<td class='tdheader'>Name</td>";
						echo "<td class='tdheader'>Unique</td>";
						echo "<td class='tdheader'>Seq. No.</td>";
						echo "<td class='tdheader'>Column #</td>";
						echo "<td class='tdheader'>Field</td>";
						echo "</tr>";
						for($i=0; $i<sizeof($result); $i++)
						{
							if($result[$i]['unique']==0)
								$unique = "no";
							else
								$unique = "yes";
	
							$query = "PRAGMA index_info(".$result[$i]['name'].")";
							$info = $db->selectArray($query);
							$span = sizeof($info);
	
							$tdWithClass = "<td class='td".($i%2 ? "1" : "2")."'>";
							$tdWithClassLeft = "<td class='td".($i%2 ? "1" : "2")."' style='text-align:left;'>";
							$tdWithClassSpan = "<td class='td".($i%2 ? "1" : "2")."' rowspan='".$span."'>";
							$tdWithClassLeftSpan = "<td class='td".($i%2 ? "1" : "2")."' style='text-align:left;' rowspan='".$span."'>";
							echo "<tr>";
							echo $tdWithClassSpan;
							echo "<a href='".PAGE."?table=".$_GET['table']."&action=index_delete&pk=".$result[$i]['name']."' style='color:red;'>delete</a>";
							echo "</td>";
							echo $tdWithClassLeftSpan;
							echo $result[$i]['name'];
							echo "</td>";
							echo $tdWithClassLeftSpan;
							echo $unique;
							echo "</td>";
							for($j=0; $j<$span; $j++)
							{
								if($j!=0)
									echo "<tr>";
								echo $tdWithClassLeft;
								echo $info[$j]['seqno'];
								echo "</td>";
								echo $tdWithClassLeft;
								echo $info[$j]['cid'];
								echo "</td>";
								echo $tdWithClassLeft;
								echo $info[$j]['name'];
								echo "</td>";
								echo "</tr>";
							}
						}
						echo "</table><br/><br/>";
					}
					
					$query = "SELECT * FROM sqlite_master WHERE type='trigger' AND tbl_name='".$_GET['table']."' ORDER BY name";
					$result = $db->selectArray($query);
					//print_r($result);
					if(sizeof($result)>0)
					{
						echo "<h2>Triggers:</h2>";
						echo "<table border='0' cellpadding='2' cellspacing='1' class='viewTable'>";
						echo "<tr>";
						echo "<td colspan='1'>";
						echo "</td>";
						echo "<td class='tdheader'>Name</td>";
						echo "<td class='tdheader'>SQL</td>";
						echo "</tr>";
						for($i=0; $i<sizeof($result); $i++)
						{
							$tdWithClass = "<td class='td".($i%2 ? "1" : "2")."'>";
							echo "<tr>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$_GET['table']."&action=trigger_delete&pk=".$result[$i]['name']."' style='color:red;'>delete</a>";
							echo "</td>";
							echo $tdWithClass;
							echo $result[$i]['name'];
							echo "</td>";
							echo $tdWithClass;
							echo $result[$i]['sql'];
							echo "</td>";
						}
						echo "</table><br/><br/>";
					}
					
					echo "<form action='".PAGE."?table=".$_GET['table']."&action=index_create' method='post'>";
					echo "<input type='hidden' name='tablename' value='".$_GET['table']."'/>";
					echo "<br/><div class='tdheader'>";
					echo "Create an index on <input type='text' name='numcolumns' style='width:30px;' value='1'/> columns <input type='submit' value='Go' name='addindex' class='btn'/>";
					echo "</div>";
					echo "</form>";
					
					echo "<form action='".PAGE."?table=".$_GET['table']."&action=trigger_create' method='post'>";
					echo "<input type='hidden' name='tablename' value='".$_GET['table']."'/>";
					echo "<br/><div class='tdheader'>";
					echo "Create a new trigger <input type='submit' value='Go' name='addindex' class='btn'/>";
					echo "</div>";
					echo "</form>";
				}
				break;
			/////////////////////////////////////////////// create column
			case "column_create":
				echo "<h2>Adding new field(s) to table '".$_POST['tablename']."'</h2>";
				if($_POST['tablefields']=="" || intval($_POST['tablefields'])<=0)
					echo "You must specify the number of table fields.";
				else if($_POST['tablename']=="")
					echo "You must specify a table name.";
				else
				{
					$num = intval($_POST['tablefields']);
					$name = $_POST['tablename'];
					echo "<form action='".PAGE."?table=".$_POST['tablename']."&action=column_create&confirm=1' method='post'>";
					echo "<input type='hidden' name='tablename' value='".$name."'/>";
					echo "<input type='hidden' name='rows' value='".$num."'/>";
					echo "<table border='0' cellpadding='2' cellspacing='1' class='viewTable'>";
					echo "<tr>";
					$headings = array("Field", "Type", "Primary Key", "Autoincrement", "Not NULL", "Default Value");
      			for($k=0; $k<count($headings); $k++)
						echo "<td class='tdheader'>" . $headings[$k] . "</td>";
					echo "</tr>";

					for($i=0; $i<$num; $i++)
					{
						$tdWithClass = "<td class='td" . ($i%2 ? "1" : "2") . "'>";
						echo "<tr>";
						echo $tdWithClass;
						echo "<input type='text' name='".$i."_field' style='width:200px;'/>";
						echo "</td>";
						echo $tdWithClass;
						echo "<select name='".$i."_type' id='".$i."_type' onchange='toggleAutoincrement(".$i.");'>";
						$types = unserialize(DATATYPES);
						for($z=0; $z<sizeof($types); $z++)
							echo "<option value='".$types[$z]."'>".$types[$z]."</option>";
						echo "</select>";
						echo "</td>";
						echo $tdWithClass;
						echo "<input type='checkbox' name='".$i."_primarykey'/> Yes";
						echo "</td>";
						echo $tdWithClass;
						echo "<input type='checkbox' name='".$i."_autoincrement' id='".$i."_autoincrement'/> Yes";
						echo "</td>";
						echo $tdWithClass;
						echo "<input type='checkbox' name='".$i."_notnull'/> Yes";
						echo "</td>";
						echo $tdWithClass;
						echo "<input type='text' name='".$i."_defaultvalue' style='width:100px;'/>";
						echo "</td>";
						echo "</tr>";
					}
					echo "<tr>";
					echo "<td class='tdheader' style='text-align:right;' colspan='6'>";
					echo "<input type='submit' value='Add Field(s)' class='btn'/> ";
					echo "<a href='".PAGE."?table=".$_POST['tablename']."&action=column_view'>Cancel</a>";
					echo "</td>";
					echo "</tr>";
					echo "</table>";
					echo "</form>";
				}
				break;
			/////////////////////////////////////////////// delete column
			case "column_delete":
				if(isset($_POST['check']))
					$pks = $_POST['check'];
				else if(isset($_GET['pk']))
					$pks = array($_GET['pk']);
				$str = $pks[0];
				$pkVal = $pks[0];
				for($i=1; $i<sizeof($pks); $i++)
				{
					$str .= ", ".$pks[$i];
					$pkVal .= ":".$pks[$i];
				}
				if($str=="") //nothing was selected so show an error
				{
					echo "<div class='confirm'>";
					echo "Error: You did not select anything.";
					echo "</div>";
					echo "<br/><br/><a href='".PAGE."?table=".$_GET['table']."&action=column_view'>Return</a>";
				}
				else
				{
					echo "<form action='".PAGE."?table=".$_GET['table']."&action=column_delete&confirm=1&pk=".$pkVal."' method='post'>";
					echo "<div class='confirm'>";
					echo "Are you sure you want to delete column(s) ".$str." from table '".$_GET['table']."'?<br/><br/>";
					echo "<input type='submit' value='Confirm' class='btn'/> ";
					echo "<a href='".PAGE."?table=".$_GET['table']."&action=column_view'>Cancel</a>";
					echo "</div>";
				}
				break;
			/////////////////////////////////////////////// edit column
			case "column_edit":
				echo "<h2>Editing column '".$_GET['pk']."' on table '".$_GET['table']."'</h2>";
				echo "Due to the limitations of SQLite, only the field name and data type can be modified.<br/><br/>";
				if(!isset($_GET['pk']))
					echo "You must specify a column.";
				else if(!isset($_GET['table']) || $_GET['table']=="")
					echo "You must specify a table name.";
				else
				{
					$query = "PRAGMA table_info('".$_GET['table']."')";
					$result = $db->selectArray($query);

					for($i=0; $i<sizeof($result); $i++)
					{
						if($result[$i][1]==$_GET['pk'])
						{
							$colVal = $result[$i][0];
							$fieldVal = $result[$i][1];
							$typeVal = $result[$i][2];
							$notnullVal = $result[$i][3];
							$defaultVal = $result[$i][4];
							$primarykeyVal = $result[$i][5];
							break;
						}
					}
					
					$name = $_GET['table'];
					echo "<form action='".PAGE."?table=".$name."&action=column_edit&confirm=1' method='post'>";
					echo "<input type='hidden' name='tablename' value='".$name."'/>";
					echo "<input type='hidden' name='oldvalue' value='".$_GET['pk']."'/>";
					echo "<table border='0' cellpadding='2' cellspacing='1' class='viewTable'>";
					echo "<tr>";
					//$headings = array("Field", "Type", "Primary Key", "Autoincrement", "Not NULL", "Default Value");
					$headings = array("Field", "Type");
      			for($k=0; $k<count($headings); $k++)
						echo "<td class='tdheader'>".$headings[$k]."</td>";
					echo "</tr>";
				
					$i = 0;
					$tdWithClass = "<td class='td" . ($i%2 ? "1" : "2") . "'>";
					echo "<tr>";
					echo $tdWithClass;
					echo "<input type='text' name='".$i."_field' style='width:200px;' value='".$fieldVal."'/>";
					echo "</td>";
					echo $tdWithClass;
					echo "<select name='".$i."_type' id='".$i."_type' onchange='toggleAutoincrement(".$i.");'>";
					$types = unserialize(DATATYPES);
					for($z=0; $z<sizeof($types); $z++)
					{
						if($types[$z]==$typeVal)
							echo "<option value='".$types[$z]."' selected='selected'>".$types[$z]."</option>";
						else
							echo "<option value='".$types[$z]."'>".$types[$z]."</option>";
					}
					echo "</select>";
					echo "</td>";
					/*
					echo $tdWithClass;
					if($primarykeyVal)
						echo "<input type='checkbox' name='".$i."_primarykey' checked='checked'/> Yes";
					else
						echo "<input type='checkbox' name='".$i."_primarykey'/> Yes";
					echo "</td>";
					echo $tdWithClass;
					if(1==2)
						echo "<input type='checkbox' name='".$i."_autoincrement' id='".$i."_autoincrement' checked='checked'/> Yes";
					else
						echo "<input type='checkbox' name='".$i."_autoincrement' id='".$i."_autoincrement'/> Yes";
					echo "</td>";
					echo $tdWithClass;
					if($notnullVal)
						echo "<input type='checkbox' name='".$i."_notnull' checked='checked'/> Yes";
					else
						echo "<input type='checkbox' name='".$i."_notnull'/> Yes";
					echo "</td>";
					echo $tdWithClass;
					echo "<input type='text' name='".$i."_defaultvalue' value='".$defaultVal."' style='width:100px;'/>";
					echo "</td>";
					*/
					echo "</tr>";

					echo "<tr>";
					echo "<td class='tdheader' style='text-align:right;' colspan='6'>";
					echo "<input type='submit' value='Save Changes' class='btn'/> ";
					echo "<a href='".PAGE."?table=".$_GET['table']."&action=column_view'>Cancel</a>";
					echo "</td>";
					echo "</tr>";
					echo "</table>";
					echo "</form>";
				}
				break;
			/////////////////////////////////////////////// delete index
			case "index_delete":
				echo "<form action='".PAGE."?table=".$_GET['table']."&action=index_delete&pk=".$_GET['pk']."&confirm=1' method='post'>";
				echo "<div class='confirm'>";
				echo "Are you sure you want to delete index '".$_GET['pk']."'?<br/><br/>";
				echo "<input type='submit' value='Confirm' class='btn'/> ";
				echo "<a href='".PAGE."?table=".$_GET['table']."&action=column_view'>Cancel</a>";
				echo "</div>";
				echo "</form>";
				break;
			/////////////////////////////////////////////// delete trigger
			case "trigger_delete":
				echo "<form action='".PAGE."?table=".$_GET['table']."&action=trigger_delete&pk=".$_GET['pk']."&confirm=1' method='post'>";
				echo "<div class='confirm'>";
				echo "Are you sure you want to delete trigger '".$_GET['pk']."'?<br/><br/>";
				echo "<input type='submit' value='Confirm' class='btn'/> ";
				echo "<a href='".PAGE."?table=".$_GET['table']."&action=column_view'>Cancel</a>";
				echo "</div>";
				echo "</form>";
				break;
			/////////////////////////////////////////////// create trigger
			case "trigger_create":
				echo "<h2>Creating new trigger on table '".$_POST['tablename']."'</h2>";
				if($_POST['tablename']=="")
					echo "You must specify a table name.";
				else
				{
					echo "<form action='".PAGE."?table=".$_POST['tablename']."&action=trigger_create&confirm=1' method='post'>";
					echo "Trigger name: <input type='text' name='trigger_name'/><br/><br/>";
					echo "<fieldset><legend>Database Event</legend>";
					echo "Before/After: ";
					echo "<select name='beforeafter'>";
					echo "<option value=''></option>";
					echo "<option value='BEFORE'>BEFORE</option>";
					echo "<option value='AFTER'>AFTER</option>";
					echo "<option value='INSTEAD OF'>INSTEAD OF</option>";
					echo "</select>";
					echo "<br/><br/>";
					echo "Event: ";
					echo "<select name='event'>";
					echo "<option value='DELETE'>DELETE</option>";
					echo "<option value='INSERT'>INSERT</option>";
					echo "<option value='UPDATE'>UPDATE</option>";
					echo "</select>";
					echo "</fieldset><br/><br/>";
					echo "<fieldset><legend>Trigger Action</legend>";
					echo "<input type='checkbox' name='foreachrow'/> For Each Row<br/><br/>";
					echo "WHEN expression (type expression without 'WHEN'):<br/>";
					echo "<textarea name='whenexpression' style='width:500px; height:100px;'></textarea>";
					echo "<br/><br/>";
					echo "Trigger Steps (semicolon terminated):<br/>";
					echo "<textarea name='triggersteps' style='width:500px; height:100px;'></textarea>";
					echo "</fieldset><br/><br/>";
					echo "<input type='submit' value='Create Trigger' class='btn'/> ";
					echo "<a href='".PAGE."?table=".$_POST['tablename']."&action=column_view'>Cancel</a>";
					echo "</form>";
				}
				break;
			/////////////////////////////////////////////// create index
			case "index_create":
				echo "<h2>Creating new index on table '".$_POST['tablename']."'</h2>";
				if($_POST['numcolumns']=="" || intval($_POST['numcolumns'])<=0)
					echo "You must specify the number of table fields.";
				else if($_POST['tablename']=="")
					echo "You must specify a table name.";
				else
				{
					echo "<form action='".PAGE."?table=".$_POST['tablename']."&action=index_create&confirm=1' method='post'>";
					$num = intval($_POST['numcolumns']);
					$query = "PRAGMA table_info('".$_POST['tablename']."')";
					$result = $db->selectArray($query);
					echo "<fieldset><legend>Define index properties</legend>";
					echo "Index name: <input type='text' name='name'/><br/>";
					echo "Duplicate values: ";
					echo "<select name='duplicate'>";
					echo "<option value='yes'>Allowed</option>";
					echo "<option value='no'>Not Allowed</option>";
					echo "</select><br/>";
					echo "</fieldset>";
					echo "<br/>";
					echo "<fieldset><legend>Define index columns</legend>";
					for($i=0; $i<$num; $i++)
					{
						echo "<select name='".$i."_field'>";
						echo "<option value=''>--Ignore--</option>";
						for($j=0; $j<sizeof($result); $j++)
							echo "<option value='".$result[$j][1]."'>".$result[$j][1]."</option>";
						echo "</select> ";
						echo "<select name='".$i."_order'>";
						echo "<option value=''></option>";
						echo "<option value=' ASC'>Ascending</option>";
						echo "<option value=' DESC'>Descending</option>";
						echo "</select><br/>";
					}
					echo "</fieldset>";
					echo "<br/><br/>";
					echo "<input type='hidden' name='num' value='".$num."'/>";
					echo "<input type='submit' value='Create Index' class='btn'/> ";
					echo "<a href='".PAGE."?table=".$_POST['tablename']."&action=column_view'>Cancel</a>";
					echo "</form>";
				}
				break;
		}
		echo "</div>";
	}
	
	$view = "structure";
		
	if(!isset($_GET['table']) && !isset($_GET['confirm']) && (!isset($_GET['action']) || (isset($_GET['action']) && $_GET['action']!="table_create"))) //the absence of these fields means we are viewing the database homepage
	{
		if(isset($_GET['view']))
			$view = $_GET['view'];
		else
			$view = "structure";

		echo "<a href='".PAGE."?view=structure' ";
		if($view=="structure")
			echo "class='tab_pressed'";
		else
			echo "class='tab'";
		echo ">Structure</a>";
		echo "<a href='".PAGE."?view=sql' ";
		if($view=="sql")
			echo "class='tab_pressed'";
		else
			echo "class='tab'";
		echo ">SQL</a>";
		echo "<a href='".PAGE."?view=export' ";
		if($view=="export")
			echo "class='tab_pressed'";
		else
			echo "class='tab'";
		echo ">Export</a>";
		echo "<a href='".PAGE."?view=import' ";
		if($view=="import")
			echo "class='tab_pressed'";
		else
			echo "class='tab'";
		echo ">Import</a>";
		echo "<a href='".PAGE."?view=vacuum' ";
		if($view=="vacuum")
			echo "class='tab_pressed'";
		else
			echo "class='tab'";
		echo ">Vacuum</a>";
		if($directory!==false && is_writable($directory))
		{
			echo "<a href='".PAGE."?view=rename' ";
			if($view=="rename")
				echo "class='tab_pressed'";
			else
				echo "class='tab'";
			echo ">Rename Database</a>";
			
			echo "<a href='".PAGE."?view=delete' style='color:red;' ";
			if($view=="delete")
				echo "class='tab_pressed'";
			else
				echo "class='tab'";
			echo ">Delete Database</a>";
		}
		echo "<div style='clear:both;'></div>";
		echo "<div id='main'>";

		if($view=="structure") //database structure - view of all the tables
		{
			$query = "SELECT sqlite_version() AS sqlite_version";
			$queryVersion = $db->select($query);
			$realVersion = $queryVersion['sqlite_version'];
			
			echo "<b>Database name</b>: ".$db->getName()."<br/>";
			echo "<b>Path to database</b>: ".$db->getPath()."<br/>";
			echo "<b>Size of database</b>: ".$db->getSize()."<br/>";
			echo "<b>Database last modified</b>: ".$db->getDate()."<br/>";
			echo "<b>SQLite version</b>: ".$realVersion."<br/>";
			echo "<b>SQLite extension</b> ".helpLink("SQLite Library Extensions").": ".$db->getType()."<br/>";
			echo "<b>PHP version</b>: ".phpversion()."<br/><br/>";
			
			if(isset($_GET['sort']))
				$_SESSION[COOKIENAME.'sort'] = $_GET['sort'];
			else
				unset($_SESSION[COOKIENAME.'sort']);
			if(isset($_GET['order']))
				$_SESSION[COOKIENAME.'order'] = $_GET['order'];
			else
				unset($_SESSION[COOKIENAME.'order']);
					
			$query = "SELECT type, name FROM sqlite_master WHERE type='table' OR type='view'";
			$queryAdd = "";
			if(isset($_SESSION[COOKIENAME.'sort']))
				$queryAdd .= " ORDER BY ".$_SESSION[COOKIENAME.'sort'];
			if(isset($_SESSION[COOKIENAME.'order']))
				$queryAdd .= " ".$_SESSION[COOKIENAME.'order'];
			$query .= $queryAdd;
			$result = $db->selectArray($query);

			$j = 0;
			for($i=0; $i<sizeof($result); $i++)
				if(substr($result[$i]['name'], 0, 7)!="sqldite_" && $result[$i]['name']!="")
					$j++;

			if($j==0)
				echo "No tables in database.<br/><br/>";
			else
			{
				echo "<table border='0' cellpadding='2' cellspacing='1' class='viewTable'>";
				echo "<tr>";
				
				echo "<td class='tdheader'>";
				echo "<a href='".PAGE."?sort=type";
				if(isset($_SESSION[COOKIENAME.'sort']))
					$orderTag = ($_SESSION[COOKIENAME.'sort']=="type" && $_SESSION[COOKIENAME.'order']=="ASC") ? "DESC" : "ASC";
				else
					$orderTag = "ASC";
				echo "&order=".$orderTag;
				echo "'>Type</a> ".helpLink("Tables vs. Views");
				if(isset($_SESSION[COOKIENAME.'sort']) && $_SESSION[COOKIENAME.'sort']=="type")
					echo (($_SESSION[COOKIENAME.'order']=="ASC") ? " <b>&uarr;</b>" : " <b>&darr;</b>");
				echo "</td>";
				
				echo "<td class='tdheader'>";
				echo "<a href='".PAGE."?sort=name";
				if(isset($_SESSION[COOKIENAME.'sort']))
					$orderTag = ($_SESSION[COOKIENAME.'sort']=="name" && $_SESSION[COOKIENAME.'order']=="ASC") ? "DESC" : "ASC";
				else
					$orderTag = "ASC";
				echo "&order=".$orderTag;
				echo "'>Name</a>";
				if(isset($_SESSION[COOKIENAME.'sort']) && $_SESSION[COOKIENAME.'sort']=="name")
					echo (($_SESSION[COOKIENAME.'order']=="ASC") ? " <b>&uarr;</b>" : " <b>&darr;</b>");
				echo "</td>";
				
				echo "<td class='tdheader' colspan='10'>Action</td>";
				echo "<td class='tdheader'>Records</td>";
				echo "</tr>";
				
				$totalRecords = 0;
				for($i=0; $i<sizeof($result); $i++)
				{
					if(substr($result[$i]['name'], 0, 7)!="sqldite_" && $result[$i]['name']!="")
					{
						$records = $db->numRows($result[$i]['name']);
						$totalRecords += $records;
						$tdWithClass = "<td class='td".($i%2 ? "1" : "2")."'>";
						$tdWithClassLeft = "<td class='td".($i%2 ? "1" : "2")."' style='text-align:left;'>";
						
						if($result[$i]['type']=="table")
						{
							echo "<tr>";
							echo $tdWithClassLeft;
							echo "Table";
							echo "</td>";
							echo $tdWithClassLeft;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=row_view'>".$result[$i]['name']."</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=row_view'>Browse</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=column_view'>Structure</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=table_sql'>SQL</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=table_search'>Search</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=row_create'>Insert</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=table_export'>Export</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=table_import'>Import</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=table_rename'>Rename</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=table_empty' style='color:red;'>Empty</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=table_drop' style='color:red;'>Drop</a>";
							echo "</td>";
							echo $tdWithClass;
							echo $records;
							echo "</td>";
							echo "</tr>";
						}
						else
						{
							echo "<tr>";
							echo $tdWithClassLeft;
							echo "View";
							echo "</td>";
							echo $tdWithClassLeft;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=row_view&view=1'>".$result[$i]['name']."</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=row_view&view=1'>Browse</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=column_view&view=1'>Structure</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=table_sql&view=1'>SQL</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=table_search&view=1'>Search</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=table_export&view=1'>Export</a>";
							echo "</td>";
							echo $tdWithClass;
							echo "";
							echo "</td>";
							echo $tdWithClass;
							echo "";
							echo "</td>";
							echo $tdWithClass;
							echo "";
							echo "</td>";
							echo $tdWithClass;
							echo "<a href='".PAGE."?table=".$result[$i]['name']."&action=view_drop&view=1' style='color:red;'>Drop</a>";
							echo "</td>";
							echo $tdWithClass;
							echo $records;
							echo "</td>";
							echo "</tr>";
						}
					}
				}
				echo "<tr>";
				echo "<td class='tdheader' colspan='12'>".sizeof($result)." total</td>";
				echo "<td class='tdheader' colspan='1' style='text-align:right;'>".$totalRecords."</td>";
				echo "</tr>";
				echo "</table>";
				echo "<br/>";
			}
			echo "<fieldset>";
			echo "<legend><b>Create new table on database '".$db->getName()."'</b></legend>";
			echo "<form action='".PAGE."?action=table_create' method='post'>";
			echo "Name: <input type='text' name='tablename' style='width:200px;'/> ";
			echo "Number of Fields: <input type='text' name='tablefields' style='width:90px;'/> ";
			echo "<input type='submit' name='createtable' value='Go' class='btn'/>";
			echo "</form>";
			echo "</fieldset>";
			echo "<br/>";
			echo "<fieldset>";
			echo "<legend><b>Create new view on database '".$db->getName()."'</b></legend>";
			echo "<form action='".PAGE."?action=view_create&confirm=1' method='post'>";
			echo "Name: <input type='text' name='viewname' style='width:200px;'/> ";
			echo "Select Statement ".helpLink("Writing a Select Statement for a New View").": <input type='text' name='select' style='width:400px;'/> ";
			echo "<input type='submit' name='createtable' value='Go' class='btn'/>";
			echo "</form>";
			echo "</fieldset>";
		}
		else if($view=="sql") //database SQL editor
		{
			$isSelect = false;
			if(isset($_POST['query']) && $_POST['query']!="")
			{
				$delimiter = $_POST['delimiter'];
				$queryStr = stripslashes($_POST['queryval']);
				$query = explode_sql($delimiter, $queryStr); //explode the query string into individual queries based on the delimiter

				for($i=0; $i<sizeof($query); $i++) //iterate through the queries exploded by the delimiter
				{
					if(str_replace(" ", "", str_replace("\n", "", str_replace("\r", "", $query[$i])))!="") //make sure this query is not an empty string
					{
						$startTime = microtime(true);
						if(strpos(strtolower($query[$i]), "select ")!==false)
						{
							$isSelect = true;
							$result = $db->selectArray($query[$i], "assoc");
						}
						else
						{
							$isSelect = false;
							$result = $db->query($query[$i]);
						}
						$endTime = microtime(true);
						$time = round(($endTime - $startTime), 4);

						echo "<div class='confirm'>";
						echo "<b>";
						// 22 August 2011: gkf fixed bugs 46, 51 and 52.
						if($result)
						{
							if($isSelect)
							{
								$affected = sizeof($result);
								echo "Showing ".$affected." row(s). ";
							}
							else
							{
								$affected = $db->getAffectedRows();
								echo $affected." row(s) affected. ";
							}
							echo "(Query took ".$time." sec)</b><br/>";
						}
						else
						{
							echo "There is a problem with the syntax of your query ";
							echo "(Query was not executed)</b><br/>";
						}
						echo "<span style='font-size:11px;'>".$query[$i]."</span>";
						echo "</div><br/>";
						if($isSelect)
						{
							if(sizeof($result)>0)
							{
								$headers = array_keys($result[0]);

								echo "<table border='0' cellpadding='2' cellspacing='1' class='viewTable'>";
								echo "<tr>";
								for($j=0; $j<sizeof($headers); $j++)
								{
									echo "<td class='tdheader'>";
									echo $headers[$j];
									echo "</td>";
								}
								echo "</tr>";
								for($j=0; $j<sizeof($result); $j++)
								{
									$tdWithClass = "<td class='td".($j%2 ? "1" : "2")."'>";
									echo "<tr>";
									for($z=0; $z<sizeof($headers); $z++)
									{
										echo $tdWithClass;
										echo $result[$j][$headers[$z]];
										echo "</td>";
									}
									echo "</tr>";
								}
								echo "</table><br/><br/>";
							}
						}
					}
				}
			}
			else
			{
				$delimiter = ";";
				$queryStr = "";
			}

			echo "<fieldset>";
			echo "<legend><b>Run SQL query/queries on database '".$db->getName()."'</b></legend>";
			echo "<form action='".PAGE."?view=sql' method='post'>";
			echo "<textarea style='width:100%; height:300px;' name='queryval'>".$queryStr."</textarea>";
			echo "Delimiter <input type='text' name='delimiter' value='".$delimiter."' style='width:50px;'/> ";
			echo "<input type='submit' name='query' value='Go' class='btn'/>";
			echo "</form>";
		}
		else if($view=="vacuum")
		{
			if(isset($_POST['vacuum']))
			{
				$query = "VACUUM";
				$db->query($query);
				echo "<div class='confirm'>";
				echo "The database, '".$db->getName()."', has been VACUUMed.";
				echo "</div><br/>";
			}
			echo "<form method='post' action='".PAGE."?view=vacuum'>";
			echo "Large databases sometimes need to be VACUUMed to reduce their footprint on the server. Click the button below to VACUUM the database, '".$db->getName()."'.";
			echo "<br/><br/>";
			echo "<input type='submit' value='VACUUM' name='vacuum' class='btn'/>";
			echo "</form>";
		}
		else if($view=="export")
		{
			echo "<form method='post' action='".PAGE."?view=export'>";
			echo "<fieldset style='float:left; width:260px; margin-right:20px;'><legend><b>Export</b></legend>";
			echo "<select multiple='multiple' size='10' style='width:240px;' name='tables[]'>";
			$query = "SELECT name FROM sqlite_master WHERE type='table' OR type='view' ORDER BY name";
			$result = $db->selectArray($query);
			for($i=0; $i<sizeof($result); $i++)
			{
				if(substr($result[$i]['name'], 0, 7)!="sqlite_" && $result[$i]['name']!="")
					echo "<option value='".$result[$i]['name']."' selected='selected'>".$result[$i]['name']."</option>";
			}
			echo "</select>";
			echo "<br/><br/>";
			echo "<input type='radio' name='export_type' checked='checked' value='sql' onclick='toggleExports(\"sql\");'/> SQL";
			echo "<br/><input type='radio' name='export_type' value='csv' onclick='toggleExports(\"csv\");'/> CSV";
			echo "</fieldset>";
			
			echo "<fieldset style='float:left; max-width:350px;' id='exportoptions_sql'><legend><b>Options</b></legend>";
			echo "<input type='checkbox' checked='checked' name='structure'/> Export with structure ".helpLink("Export Structure to SQL File")."<br/>";
			echo "<input type='checkbox' checked='checked' name='data'/> Export with data ".helpLink("Export Data to SQL File")."<br/>";
			echo "<input type='checkbox' name='drop'/> Add DROP TABLE ".helpLink("Add Drop Table to Exported SQL File")."<br/>";
			echo "<input type='checkbox' checked='checked' name='transaction'/> Add TRANSACTION ".helpLink("Add Transaction to Exported SQL File")."<br/>";
			echo "<input type='checkbox' checked='checked' name='comments'/> Comments ".helpLink("Add Comments to Exported SQL File")."<br/>";
			echo "</fieldset>";
			
			echo "<fieldset style='float:left; max-width:350px; display:none;' id='exportoptions_csv'><legend><b>Options</b></legend>";
			echo "<div style='float:left;'>Fields terminated by</div>";
			echo "<input type='text' value=';' name='export_csv_fieldsterminated' style='float:right;'/>";
			echo "<div style='clear:both;'>";
			echo "<div style='float:left;'>Fields enclosed by</div>";
			echo "<input type='text' value='\"' name='export_csv_fieldsenclosed' style='float:right;'/>";
			echo "<div style='clear:both;'>";
			echo "<div style='float:left;'>Fields escaped by</div>";
			echo "<input type='text' value='\' name='export_csv_fieldsescaped' style='float:right;'/>";
			echo "<div style='clear:both;'>";
			echo "<div style='float:left;'>Replace NULL by</div>";
			echo "<input type='text' value='NULL' name='export_csv_replacenull' style='float:right;'/>";
			echo "<div style='clear:both;'>";
			echo "<input type='checkbox' name='export_csv_crlf'/> Remove CRLF characters within fields<br/>";
			echo "<input type='checkbox' checked='checked' name='export_csv_fieldnames'/> Put field names in first row";
			echo "</fieldset>";
			
			echo "<div style='clear:both;'></div>";
			echo "<br/><br/>";
			echo "<fieldset style='float:left;'><legend><b>Save As</b></legend>";
			echo "<input type='hidden' name='database_num' value='".$_SESSION[COOKIENAME.'currentDB']."'/>";
			$file = pathinfo($db->getPath());
			$name = $file['filename'];
			echo "<input type='text' name='filename' value='".$name.".".date("n-j-y").".dump' style='width:400px;'/> <input type='submit' name='export' value='Export' class='btn'/>";
			echo "</fieldset>";
			echo "</form>";
		}
		else if($view=="import")
		{
			if(isset($_POST['import']))
			{
				echo "<div class='confirm'>";
				if($importSuccess===true)
					echo "Import was successful.";
				else
					echo $importSuccess;
				echo "</div><br/>";
			}
			
			echo "<form method='post' action='".PAGE."?view=import' enctype='multipart/form-data'>";
			echo "<fieldset style='float:left; width:260px; margin-right:20px;'><legend><b>Import</b></legend>";
			echo "<input type='radio' name='import_type' checked='checked' value='sql' onclick='toggleImports(\"sql\");'/> SQL";
			echo "<br/><input type='radio' name='import_type' value='csv' onclick='toggleImports(\"csv\");'/> CSV";
			echo "</fieldset>";
			
			echo "<fieldset style='float:left; max-width:350px;' id='importoptions_sql'><legend><b>Options</b></legend>";
			echo "No options";
			echo "</fieldset>";
			
			echo "<fieldset style='float:left; max-width:350px; display:none;' id='importoptions_csv'><legend><b>Options</b></legend>";
			echo "<div style='float:left;'>Table that CSV pertains to</div>";
			echo "<select name='single_table' style='float:right;'>";
			$query = "SELECT name FROM sqlite_master WHERE type='table' OR type='view' ORDER BY name";
			$result = $db->selectArray($query);
			for($i=0; $i<sizeof($result); $i++)
			{
				if(substr($result[$i]['name'], 0, 7)!="sqlite_" && $result[$i]['name']!="")
					echo "<option value='".$result[$i]['name']."'>".$result[$i]['name']."</option>";
			}
			echo "</select>";
			echo "<div style='clear:both;'>";
			echo "<div style='float:left;'>Fields terminated by</div>";
			echo "<input type='text' value=';' name='import_csv_fieldsterminated' style='float:right;'/>";
			echo "<div style='clear:both;'>";
			echo "<div style='float:left;'>Fields enclosed by</div>";
			echo "<input type='text' value='\"' name='import_csv_fieldsenclosed' style='float:right;'/>";
			echo "<div style='clear:both;'>";
			echo "<div style='float:left;'>Fields escaped by</div>";
			echo "<input type='text' value='\' name='import_csv_fieldsescaped' style='float:right;'/>";
			echo "<div style='clear:both;'>";
			echo "<div style='float:left;'>NULL represented by</div>";
			echo "<input type='text' value='NULL' name='import_csv_replacenull' style='float:right;'/>";
			echo "<div style='clear:both;'>";
			echo "<input type='checkbox' checked='checked' name='import_csv_fieldnames'/> Field names in first row";
			echo "</fieldset>";
			
			echo "<div style='clear:both;'></div>";
			echo "<br/><br/>";
			
			echo "<fieldset><legend><b>File to import</b></legend>";
			echo "<input type='file' value='Choose File' name='file' style='background-color:transparent; border-style:none;'/> <input type='submit' value='Import' name='import' class='btn'/>";
			echo "</fieldset>";
		}
		else if($view=="rename")
		{
			if(isset($dbexists))
			{
				echo "<div class='confirm'>";
				if($oldpath==$newpath)
					echo "Error: You didn't change the value dumbass.";
				else
					echo "Error: A database of the name '".$newpath."' already exists.";
				echo "</div><br/>";
			}
			if(isset($justrenamed))
			{
				echo "<div class='confirm'>";
				echo "Database '".$oldpath."' has been renamed to '".$newpath."'.";
				echo "</div><br/>";
			}
			echo "<form action='".PAGE."?view=rename&database_rename=1' method='post'>";
			echo "<input type='hidden' name='oldname' value='".$db->getPath()."'/>";
			echo "Rename database '".$db->getPath()."' to <input type='text' name='newname' style='width:200px;' value='".$db->getPath()."'/> <input type='submit' value='Rename' name='rename' class='btn'/>";
			echo "</form>";	
		}
		else if($view=="delete")
		{
			echo "<form action='".PAGE."?database_delete=1' method='post'>";
			echo "<div class='confirm'>";
			echo "Are you sure you want to delete the database '".$db->getPath()."'?<br/><br/>";
			echo "<input name='database_delete' value='".$db->getPath()."' type='hidden'/>";
			echo "<input type='submit' value='Confirm' class='btn'/> ";
			echo "<a href='".PAGE."'>Cancel</a>";
			echo "</div>";
			echo "</form>";	
		}

		echo "</div>";
	}

	echo "<br/>";
	$endTimeTot = microtime(true); //get the current time at this point in the execution
	$timeTot = round(($endTimeTot - $startTimeTot), 4); //calculate the total time for page load
	echo "<span style='font-size:11px;'>Powered by <a href='http://code.google.com/p/phpliteadmin/' target='_blank' style='font-size:11px;'>".PROJECT."</a> | Page generated in ".$timeTot." seconds.</span>";
	echo "</div>";
	echo "</div>";
	$db->close(); //close the database
}
echo "</body>";
echo "</html>";

?>