<?php


$allow_edit = false;

ini_set('error_reporting', E_ALL);
include_once("header.php");
$years = array("2007","2008","2009","2010","2011","2012","2013","2014");


foreach($_GET as $key =>$value){ $_GET[$key] = addslashes($value); }
foreach($_POST as $key =>$value){ $_POST[$key] = addslashes($value); }


$year = "";
$yearSQLAndPublication = "";
$yearSQLFull = "";

$yearSQLAndNote = "";
$yearSQLFullNote = "";

if (isset($_GET['year'])){
	$year = $_GET['year'];

	$yearSQLAndPublication = "AND publication.year = ".$year;
	$yearSQLFull = "WHERE publication.year = ".$year;

	$yearSQLAndNote = "AND note.year = ".$year;
	$yearSQLFullNote = "WHERE note.year = ".$year;
}

?>
<center>
	<h1>Publication / Author / Institute</h1>
	
</center>

<ul class="nav nav-tabs">
  <li <?php if(!isset($_GET['page']) || $_GET['page'] == "publications"){echo "class='active'";}?>><a href="?page=publications">Publications</a></li>
  <li <?php if($_GET['page'] == "authors"){echo "class='active'";}?>><a href="?page=authors">Authors</a></li>
  <li <?php if($_GET['page'] == "institutes"){echo "class='active'";}?>><a href="?page=institutes">Institutes</a></li>
  <li <?php if($_GET['page'] == "countries"){echo "class='active'";}?>><a href="?page=countries">Countries</a></li>
<li <?php if($_GET['page'] == "notes"){echo "class='active'";}?>><a href="?page=notes">Notes</a></li>
  <li <?php if($_GET['page'] == "search"){echo "class='active'";}?>><a href="?page=search"></a></li>
</ul>

 Year:
<select name="year" onchange="changeYear(this.value);">
<option value="">all</option>
<?php
	foreach($years as $ayear){
		$selected = "";
		if ($ayear == $year){ $selected = "SELECTED"; }
		echo "<option value='".$ayear."' ".$selected.">".$ayear."</option>";
	}
?>
</select>

<?php





try
  { 
    $db = new PDO('sqlite:aq.db');


function getAuthorsByPublicationId($publication_id){
	return $db->query('SELECT author.id, author.fullname, author.email FROM author, publication_author WHERE publication_author.author_id = author.id AND publication_author.publication_id = "'.$publication_id.'"')->fetchAll();	
}

function getInstitute(){
	return $db->query('SELECT * FROM institute ORDER BY name ASC')->fetchAll();	
}



switch($_GET['page']){

case "note":
break;

case "notes":

	$notes = $db->query('SELECT note.id as id, note.code as code, note.name as name, author.id as aid, author.fullname as afullname, institute.id as iid, institute.name as iname, country.id as cid, country.name as cname 
			FROM author, note, institute, country, note_country, note_author, note_institute 
			WHERE note_country.country_id = country.id AND note_country.note_id = note.id AND
			      note_author.author_id = author.id AND note_author.note_id = note.id AND
			      note_institute.institute_id = institute.id AND note_institute.note_id = note.id '.$yearSQLAndNote.'	
			ORDER BY note.code ASC')->fetchAll();	

	echo "<br/>Count of notes: <strong>".sizeof($notes)."</strong><br/><br/>";

	echo "<table class='table table-hover'>";
        echo "<tr><th width='150px'>Code</th><th>Publication</th><th>Title</th><th>Country</th><th>Institute</th><th width='200px'>Submitter</th></tr>";

	foreach($notes as $note){		
		echo "<tr>
			<td>
				<a class='btn btn-mini btn-warning' target='_blank' href='http://cms.cern.ch/iCMS/jsp/db_notes/showNoteDetails.jsp?noteID=".$note['code']."'>".$note['code']."</a>
			</td>
			<td>";
			$publications = $db->query("SELECT publication.* FROM publication, note_publication WHERE note_publication.publication_id = publication.id AND note_publication.note_id='".$note['id']."'")->fetchAll();
			foreach($publications as $publication){
				echo "<a class='btn btn-mini btn-primary' href='?page=publication&id=".$publication['id']."'>".$publication['code']."</a> ";
			}
				
			echo "</td>
			<td>
				".$note['name']."
			</td>
			<td>
				<a class='btn btn-mini ' href='?page=country&id=".$note['cid']."'>".$note['cname']."</a>
			</td>
			<td>
				<a class='btn btn-mini btn-info' href='?page=institute&id=".$note['iid']."'>".$note['iname']."</a>
			</td>
			<td>
				<a class='btn btn-mini btn-info' href='?page=author&id=".$note['aid']."'>".$note['afullname']."</a>
			</td>
		  </tr>";
	}
	echo "</table>";

break;

case "author":
   if (is_numeric($_GET['id'])){
	$author = $db->query('SELECT * FROM author WHERE id='.$_GET['id'])->fetch();			
	
	echo "<center><h2>Author: ".$author['fullname']."</h2>";
	echo "Email: ".$author['email']."<br/>";
	echo "Cadi URL: <a href='".$author['cadi_url']."'>here</a><br/>";
        if ($allow_edit){
		echo '<a href="#" id="edit" class="btn btn-small btn-success" data-toggle="popover" title="Author info" data-content="';
		echo "<form action='edit.php?action=editAuthor' method='POST'>
		<input type='hidden' name='return' value='author'>
		<input type='hidden' name='rid' value='".$author['id']."'>
		<input type='hidden' name='author_id' value='".$author['id']."'>
		Fullname<br/>
		<input type='text' name='fullname' value='".$author['fullname']."'>
		Email<br/>
		<input type='text' name='email' value='".$author['email']."'>
		CADI url<br/>
		<input type='text' name='cadi_url' value='".$author['cadi_url']."'>
		<input type='submit' value='Save changes' class='btn btn-small btn-success'>
		</form>";
        	echo '">edit</a>';
        }
	echo "<hr/><h4>Institutes</h4></center>";

        echo "<table class='table table-hover'>";
	echo "<tr><th>Name</th><th></th></tr>";
        $institutes = $db->query('SELECT institute.id, institute.name, author_institute.id as ai_id FROM institute, author_institute WHERE author_institute.institute_id = institute.id AND author_institute.author_id = "'.$author['id'].'"')->fetchAll();	
	foreach($institutes as $institute){		
		echo "<tr>
			<td><a class='btn btn-mini btn-info' href='?page=institute&id=".$institute['id']."'>".$institute['name']."</a> 
			</td>
			<td>";
		if ($allow_edit){
			echo "<a href='edit.php?action=removeAuthorInstitute&ai_id=".$institute['ai_id']."&rid=".$author['id']."&return=author' class='btn btn-small btn-danger'>Resign institute</a>";
		}
		echo "</td>
		      </tr>";
	}	
	echo "</table>";
        if ($allow_edit){
		echo "<hr/><center><h4>Assign institute</h4>";

		echo "<form action='edit.php?action=addAuthorInstitute' method='POST'>";
		echo "<input type='hidden' name='author_id' value='".$author['id']."'>";
		echo "<input type='hidden' name='rid' value='".$author['id']."'>";
		echo "<input type='hidden' name='return' value='author'>";
		echo "<select name='institute_id'>";
		$available_institutes = $db->query('SELECT * FROM institute ORDER BY name ASC')->fetchAll();	

		foreach($available_institutes as $available_institute){
			echo "<option value='".$available_institute['id']."'>".$available_institute['name']."</option>";
		}	
		echo "</select><br/>";
		echo "<input type='submit' value='Assign institute' class='btn btn-small btn-success'>";
		echo "</form></center>";  
	}

        echo "<center><hr/><h4>Publications</h4></center>";

        echo "<table class='table table-hover'>";
	echo "<tr><th>Code</th><th>Title</th><th></th></tr>";
        $publications = $db->query('SELECT publication.id, publication.code, publication.title, publication_author.id as pa_id FROM publication, publication_author WHERE publication_author.publication_id = publication.id AND publication_author.author_id = "'.$author['id'].'" '.$yearSQLAndPublication)->fetchAll();	
	foreach($publications as $publication){		
		echo "<tr>
			<td>
				<a class='btn btn-mini btn-primary' href='?page=publication&id=".$publication['id']."'>".$publication['code']."</a>
			</td>
			<td>
				".$publication['title']."
			</td>
			<td>";
		if ($allow_edit){
			echo "<a href='edit.php?action=removePublicationAuthor&pa_id=".$publication['pa_id']."&rid=".$author['id']."&return=author' class='btn btn-small btn-danger'>Resign publication</a>";
		}
		echo "	</td>
		      </tr>";
	}	
	echo "</table>";
        if ($allow_edit){
		echo "<hr/><center><h4>Assign publication</h4>";

		echo "<form action='edit.php?action=addPublicationAuthor' method='POST'>";
		echo "<input type='hidden' name='author_id' value='".$author['id']."'>";
		echo "<input type='hidden' name='rid' value='".$author['id']."'>";
		echo "<input type='hidden' name='return' value='author'>";
		echo "<select name='publication_id'>";
		$available_publications = $db->query('SELECT * FROM publication ORDER BY code ASC')->fetchAll();	

		foreach($available_publications as $available_publication){
			echo "<option value='".$available_publication['id']."'>".$available_publication['code']."</option>";
		}	
		echo "</select><br/>";
		echo "<input type='submit' value='Assign publication' class='btn btn-small btn-success'>";
		echo "</form></center>";
	}

	$notes = $db->query('SELECT note.code FROM note, note_author WHERE note.id = note_author.note_id AND note_author.author_id = "'.$author['id'].'" '.$yearSQLAndNote.' ORDER BY note.code ASC')->fetchAll();		
	echo "<hr/><center><h4>".sizeof($notes)." notes wrote by this authors</h4></center>"; 
	foreach($notes as $note){
		echo "<a class='btn btn-mini btn-warning' target='_blank' href='http://cms.cern.ch/iCMS/jsp/db_notes/showNoteDetails.jsp?noteID=".$note['code']."'>".$note['code']."</a> ";
	}  
  }	
break;

case "authors":

	$authors = $db->query('SELECT DISTINCT author.* FROM author, publication, publication_author WHERE author.id = publication_author.author_id AND publication.id = publication_author.publication_id '.$yearSQLAndPublication.' ORDER BY fullname ASC')->fetchAll();

	echo "<br/>Count of authors: <strong>".sizeof($authors)."</strong><br/><br/>";	

	echo "<table class='table table-hover'>";
        echo "<tr><th>Fullname</th><th>Email</th><th>Institute</th><th>Publications</th><th>Notes</th></tr>";

	foreach($authors as $author){		
		echo "<tr><td><a class='btn btn-mini btn-inverse' href='?page=author&id=".$author['id']."'>".$author['fullname']."</a></td><td>".str_replace("@","[at]",$author['email'])."</td><td>";

		$institutes = $db->query('SELECT institute.id, institute.name FROM institute, author_institute WHERE author_institute.institute_id = institute.id AND author_institute.author_id = "'.$author['id'].'"')->fetchAll();	
		foreach($institutes as $institute){		
			echo "<a class='btn btn-mini btn-info' href='?page=institute&id=".$institute['id']."'>".$institute['name']."</a> ";
		}		
		echo "</td><td>";

                $publications = $db->query('SELECT publication.id, publication.code FROM publication, publication_author WHERE publication_author.publication_id = publication.id AND publication_author.author_id = "'.$author['id'].'" '.$yearSQLAndPublication)->fetchAll();	
		foreach($publications as $publication){		
			echo "<a class='btn btn-mini btn-primary' href='?page=publication&id=".$publication['id']."'>".$publication['code']."</a> ";
		}

		echo "</td><td>";

                $notes = $db->query('SELECT note.code FROM note, note_author WHERE note.id = note_author.note_id AND note_author.author_id = "'.$author['id'].'" '.$yearSQLAndNote.' ORDER BY note.code ASC')->fetchAll();		
		foreach($notes as $note){
			echo "<a class='btn btn-mini btn-warning' target='_blank' href='http://cms.cern.ch/iCMS/jsp/db_notes/showNoteDetails.jsp?noteID=".$note['code']."'>".$note['code']."</a> ";
		}

		echo "</td></tr>";
	}
	echo "</table>";
        if ($allow_edit){
		echo "<hr/><center><h4>Create new author</h4>";	
		echo "<form action='edit.php?action=newAuthor' method='POST'>";
		echo "<input type='text' name='fullname' placeholder='full name'><br/>";
		echo "<input type='text' name='email' placeholder='email'><br/>";
		echo "<input type='text' name='cadi_url' placeholder='CADI url'><br/>";
		echo "<input type='submit' value='create' class='btn btn-small btn-success'>";
		echo "</form></center>";
	}
break;

case "institute":
   if (is_numeric($_GET['id'])){

	$institute = $db->query('SELECT institute.id as iid, institute.name as iname, institute.cadi_url, country.id as cid, country.name as cname FROM institute, country, country_institute WHERE country_institute.institute_id = institute.id AND country_institute.country_id = country.id AND institute.id = '.$_GET['id'])->fetch();	

	echo "<center><h2>Institute: ".$institute['iname']."</h2>";
	echo "<a class='btn btn-mini ' href='?page=country&id=".$institute['cid']."'>".$institute['cname']."</a><br/>";
	echo "Cadi URL: <a targer='_blank' href='".$institute['cadi_url']."'>here</a><br/>";
        if ($allow_edit){
		echo '<a href="#" id="edit" class="btn btn-small btn-success" data-toggle="popover" title="Institute info" data-content="';
		echo "<form action='edit.php?action=editInstitute' method='POST'>
		<input type='hidden' name='return' value='institute'>
		<input type='hidden' name='rid' value='".$institute['iid']."'>
		<input type='hidden' name='institute_id' value='".$institute['iid']."'>
		Name<br/>
		<input type='text' name='name' value='".$institute['iname']."'>
		CADI url<br/>
		<input type='text' name='cadi_url' value='".$institute['cadi_url']."'>
		<input type='submit' value='Save changes' class='btn btn-small btn-success'>
		</form>";
        	echo '">edit</a>';
        }
	// List of authors

	$authors = $db->query('SELECT author.id, author.fullname, author.email, author_institute.id as ai_id FROM author, author_institute WHERE author_institute.author_id = author.id AND author_institute.institute_id = "'.$institute['iid'].'"')->fetchAll();	

        echo "<hr/><h4>Authors (".sizeof($authors).")</h4></center>";
	echo "<table class='table table-hover'>";
        echo "<tr><th>Fullname</th><th>Email</th><th></th></tr>";
	
	foreach($authors as $author){		
	    echo "<tr>
			<td>
				<a class='btn btn-mini btn-inverse' href='?page=author&id=".$author['id']."'>".$author['fullname']."</a>
			</td>
		    	<td>
				".$author['email']."
			</td>
			<td>";
		if ($allow_edit){
			echo "<a href='edit.php?action=removeAuthorInstitute&ai_id=".$author['ai_id']."&rid=".$institute['iid']."&return=institute' class='btn btn-small btn-danger'>Resign author</a>";
		}
		echo "	</td>
	 	 </tr>";	
	}	
	echo "</table>";

        if ($allow_edit){
		echo "<hr/><center><h4>Assign author</h4>";	
		echo "<form action='edit.php?action=addAuthorInstitute' method='POST'>";
		echo "<input type='hidden' name='institute_id' value='".$institute['iid']."'>";
		echo "<input type='hidden' name='return' value='institute'>";
		echo "<input type='hidden' name='rid' value='".$institute['iid']."'>";
		echo "<select name='author_id'>";
		$available_authors = $db->query('SELECT * FROM author ORDER BY fullname ASC')->fetchAll();	
	
		foreach($available_authors as $available_author){
			echo "<option value='".$available_author['id']."'>".$available_author['fullname']."</option>";
		}	
		echo "</select><br/>";
		echo "<input type='submit' value='Assign author' class='btn btn-small btn-success'>";
		echo "</form></center>"; 
	}
		
	$publications = $db->query('SELECT publication.id, publication.code FROM author, author_institute, publication, publication_author WHERE publication.id = publication_author.publication_id AND publication_author.author_id = author.id AND author_institute.author_id = author.id AND author_institute.institute_id = "'.$institute['iid'].'" '.$yearSQLAndPublication.' ORDER BY publication.year ASC')->fetchAll();		
	echo "<hr/><center><h4>".sizeof($publications)." publications wrote by authors who belong to this institute</h4></center>"; 
	foreach($publications as $publication){
		echo "<a class='btn btn-mini btn-primary' href='?page=publication&id=".$publication['id']."'>".$publication['code']."</a>, ";
	}

	$notes = $db->query('SELECT note.code FROM note, note_institute WHERE note.id = note_institute.note_id AND note_institute.institute_id = "'.$institute['iid'].'" ORDER BY note.code ASC')->fetchAll();		
	echo "<hr/><center><h4>".sizeof($notes)." notes wrote by authors who belong to this institute</h4></center>"; 
	foreach($notes as $note){
		echo "<a class='btn btn-mini btn-warning' target='_blank' href='http://cms.cern.ch/iCMS/jsp/db_notes/showNoteDetails.jsp?noteID=".$note['code']."'>".$note['code']."</a>, ";
	}
   }

break;

case "institutes":

	$institutes = $db->query('SELECT institute.id as iid, institute.name as iname, country.id as cid, country.name as cname FROM institute, country, country_institute WHERE country_institute.institute_id = institute.id AND country_institute.country_id = country.id ORDER BY institute.name ASC')->fetchAll();	

	echo "<br/>Count of institutes: <strong>".sizeof($institutes)."</strong><br/><br/>";

	echo "<table class='table table-hover'>";
        echo "<tr><th>Name</th><th>Country</th><th>Authors</th></tr>";

	foreach($institutes as $institute){		
		echo "<tr>
			<td>
				<a class='btn btn-mini btn-info' href='?page=institute&id=".$institute['iid']."'>".$institute['iname']."</a>
			</td>
			<td>
				<a class='btn btn-mini ' href='?page=country&id=".$institute['cid']."'>".$institute['cname']."</a>
			</td>
			<td>";

		$authors = $db->query('SELECT author.id, author.fullname, author.email FROM author, author_institute WHERE author_institute.author_id = author.id AND author_institute.institute_id = "'.$institute['iid'].'"')->fetchAll();	
		foreach($authors as $author){		
			echo "<a class='btn btn-mini btn-inverse' href='?page=author&id=".$author['id']."'>".$author['fullname']."</a> ";
		}				

		echo "</td></tr>";
	}
	echo "</table>";
        if ($allow_edit){
		echo "<hr/><center><h4>Create new institute</h4>";	
		echo "<form action='edit.php?action=newInstitute' method='POST'>";
		echo "<input type='text' name='name' placeholder='name'><br/>";
		echo "<input type='text' name='cadi_url' placeholder='CADI url'><br/>";
		echo "<input type='submit' value='create' class='btn btn-small btn-success'>";
		echo "</form></center>";
	}
break;

case "country":
  if (is_numeric($_GET['id'])){

	$country = $db->query('SELECT * FROM country WHERE id='.$_GET['id'])->fetch();	

	echo "<center><h2>Country: ".$country['name']."</h2>";

	
		foreach($institutes as $institute){		
			echo "<a class='btn btn-mini btn-info' href='?page=institute&id=".$institute['id']."'>".$institute['name']."</a> ";
		}


	echo "<hr/><h4>Insitutes</h4></center>";

        echo "<table class='table table-hover'>";
        echo "<tr><th>Name</th><th></th></tr>";

	$institutes = $db->query('SELECT institute.id, institute.name, country_institute.id as ci_id FROM country, country_institute, institute WHERE country_institute.institute_id = institute.id AND country_institute.country_id = country.id AND country_institute.country_id = "'.$country['id'].'"')->fetchAll();	
	foreach($institutes as $institute){		
		echo "<tr>
			<td>
				<a class='btn btn-mini btn-info' href='?page=institute&id=".$institute['id']."'>".$institute['name']."</a>
			</td>			
			<td>";
		if ($allow_edit){
			echo "<a href='edit.php?action=removeCountryInstitute&pa_id=".$institute['ci_id']."&rid=".$country['id']."&return=country' class='btn btn-small btn-danger'>Resign institute</a>";
		}
		echo "	</td>
		      </tr>";
	}		

	echo "</table>";

	$authors = $db->query('SELECT DISTINCT author.id, author.fullname, author.email FROM author, author_institute, country_institute, publication_author, publication WHERE
								publication_author.author_id = author.id AND 
								publication_author.publication_id = publication.id AND
								author_institute.author_id = author.id AND 
								author_institute.institute_id = country_institute.institute_id AND
								country_institute.country_id = "'.$country['id'].'"
								 '.$yearSQLAndPublication.' ')->fetchAll();	

	echo "<hr/><center><h4>Authors</h4></center>";
	echo "<br/>Count of authors: <strong>".sizeof($authors)."</strong><br/><br/>";	

        echo "<table class='table table-hover'>";
        echo "<tr><th>Fullname</th><th>Email</th></tr>";

	foreach($authors as $author){		
		echo "<tr>
			<td>
				<a class='btn btn-mini btn-inverse' href='?page=author&id=".$author['id']."'>".$author['fullname']."</a>
			</td>
			<td>
				".$author['email']."
			</td>			
		      </tr>";
	}
	echo "</table>";

        $notes = $db->query('SELECT note.code FROM note, note_country WHERE note.id = note_country.note_id AND note_country.country_id = "'.$country['id'].'"  '.$yearSQLAndNote.' ORDER BY note.code ASC')->fetchAll();		
	echo "<hr/><center><h4>".sizeof($notes)." notes </h4></center>"; 
	foreach($notes as $note){
		echo "<a class='btn btn-mini btn-warning' target='_blank' href='http://cms.cern.ch/iCMS/jsp/db_notes/showNoteDetails.jsp?noteID=".$note['code']."'>".$note['code']."</a>, ";
	}
  }

break;

case "countries":

	$countries = $db->query('SELECT * FROM country ORDER BY name ASC')->fetchAll();

	echo "<br/>Count of countries: <strong>".sizeof($countries)."</strong><br/><br/>";	

	echo "<table class='table table-hover'>";
        echo "<tr><th>Name</th><th>Institutes</th></tr>";

	foreach($countries as $country){		
		echo "<tr><td><a class='btn btn-mini' href='?page=country&id=".$country['id']."'>".$country['name']."</a></td><td>";

		$institutes = $db->query('SELECT institute.id, institute.name FROM country, country_institute, institute WHERE country_institute.institute_id = institute.id AND country_institute.country_id = country.id AND country_institute.country_id = "'.$country['id'].'" ORDER BY institute.name ASC')->fetchAll();	
		foreach($institutes as $institute){		
			echo "<a class='btn btn-mini btn-info' href='?page=institute&id=".$institute['id']."'>".$institute['name']."</a> ";
		}		

		echo "</td></tr>";
	}
	echo "</table></center>";

break;

case "search":

	echo "search";	

break;

case "publication":
  if (is_numeric($_GET['id'])){

	$publication = $db->query('SELECT * FROM publication WHERE id='.$_GET['id'])->fetch();	

	echo "<center><h2>Publication: ".$publication['title']."</h2>";
	echo "Code: ".$publication['code']."<br/>";
	echo "Date Created: ".$publication['date_created']."<br/>";
	echo "Cadi URL: <a href='".$publication['cadi_url']."'>here</a><br/>";
        if ($allow_edit){
		echo '<a href="#" id="edit" class="btn btn-small btn-success" data-toggle="popover" title="Publication info" data-content="';
		echo "<form action='edit.php?action=editPublication' method='POST'>
		<input type='hidden' name='return' value='publication'>
		<input type='hidden' name='rid' value='".$publication['id']."'>
		<input type='hidden' name='publication_id' value='".$publication['id']."'>
		Title<br/>
		<input type='text' name='title' value='".$publication['title']."'>
		Code<br/>
		<input type='text' name='code' value='".$publication['code']."'>
		Date created<br/>
		<input type='text' name='date_created' value='".$publication['date_created']."'>
		CADI url<br/>
		<input type='text' name='cadi_url' value='".$publication['cadi_url']."'>
		<input type='submit' value='Save changes' class='btn btn-small btn-success'>
		</form>";
	
		echo '">edit</a>';
	}

        echo "<hr/><h4>Authors</h4></center>";

        echo "<table class='table table-hover'>";
        echo "<tr><th>Fullname</th><th>Email</th><th></th></tr>";

	$authors = $db->query('SELECT author.id, author.fullname, author.email, publication_author.id as pa_id FROM author, publication_author WHERE publication_author.author_id = author.id AND publication_author.publication_id = "'.$publication['id'].'"')->fetchAll();	
	foreach($authors as $author){		
		echo "<tr>
			<td>
				<a class='btn btn-mini btn-inverse' href='?page=author&id=".$author['id']."'>".$author['fullname']."</a>
			</td>
			<td>
				".$author['email']."
			</td>
			<td>";
		if ($allow_edit){
			echo "<a href='edit.php?action=removePublicationAuthor&pa_id=".$author['pa_id']."&rid=".$publication['id']."&return=publication' class='btn btn-small btn-danger'>Resign author</a>";
		}
		echo "	</td>
		      </tr>";
	}		

	echo "</table>";
        if ($allow_edit){
		echo "<hr/><center><h4>Assign author</h4>";
	
		echo "<form action='edit.php?action=addPublicationAuthor' method='POST'>";
		echo "<input type='hidden' name='publication_id' value='".$publication['id']."'>";
		echo "<input type='hidden' name='return' value='publication'>";
		echo "<input type='hidden' name='rid' value='".$publication['id']."'>";
	
		echo "<select name='author_id'>";
		$available_authors = $db->query('SELECT * FROM author ORDER BY fullname ASC')->fetchAll();	
	
		foreach($available_authors as $available_author){
			echo "<option value='".$available_author['id']."'>".$available_author['fullname']."</option>";
		}	
		echo "</select><br/>";
		echo "<input type='submit' value='Assign author' class='btn btn-small btn-success'>";
		echo "</form></center>";
	}  
    }	
break;

case "publications":
default:

	$publications = $db->query('SELECT * FROM publication '.$yearSQLFull.' ORDER BY year DESC')->fetchAll();

	echo "<br/>Count of publications: <strong>".sizeof($publications)."</strong><br/><br/>";	          

	echo "<table class='table table-hover'>";
        echo "<tr><th width='100px'>Code</th><th width='200px'>Notes</th><th width='500px'>Title</th><th width='100px'>Date Created</th><th width='100px'>Authors</th></tr>";

	foreach($publications as $publication){		
		echo "<tr>
			<td>
				<a class='btn btn-mini btn-primary' href='?page=publication&id=".$publication['id']."'>".$publication['code']."</a>
			</td>
			<td>";
			$notes = $db->query('SELECT note.code FROM note, note_publication WHERE note_publication.note_id = note.id AND note_publication.publication_id = "'.$publication['id'].'" ORDER BY note.code DESC')->fetchAll();
			foreach ($notes as $note){
				echo "<a class='btn btn-mini btn-warning' target='_blank' href='http://cms.cern.ch/iCMS/jsp/db_notes/showNoteDetails.jsp?noteID=".$note['code']."'>".$note['code']."</a> ";
			}
		echo"	</td>
			<td>
				".$publication['title']."
			</td>
			<td>
				".$publication['date_created']."
			</td><td>";

		$authors = $db->query('SELECT author.id, author.fullname, author.email FROM author, publication_author WHERE publication_author.author_id = author.id AND publication_author.publication_id = "'.$publication['id'].'"')->fetchAll();	
		foreach($authors as $author){		
			echo "<a class='btn btn-mini btn-inverse' href='?page=author&id=".$author['id']."'>".$author['fullname']."</a> ";
		}		
		echo "</td></tr>";
	}
	echo "</table>";
        if ($allow_edit){
		echo "<hr/><center><h4>Create new Publication</h4>";
		echo "<form action='edit.php?action=newPublication' method='POST'>";
		echo "<input type='text' name='title' placeholder='title'><br/>";
		echo "<input type='text' name='code' placeholder='code'><br/>";
		echo "<input type='text' name='date_created' placeholder='date created (YYYY/MM/DD)'><br/>";
		echo "<input type='text' name='cadi_url' placeholder='CADI url'><br/>";
		echo "<input type='submit' value='create' class='btn btn-small btn-success'>";
		echo "</form></center>";
	}
}

?>

<?php
}
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }
?>
<hr/>
Color info 
<a class='btn btn-mini btn-primary' href="#">Publication</a>
<a class='btn btn-mini btn-inverse' href="#">Authors</a>
<a class='btn btn-mini btn-info' href="#">Institute</a>
<a class='btn btn-mini' href="#">Country</a>
<a class='btn btn-mini btn-warning' href="#">Note</a>
<?php
include_once("footer.php");
?>

