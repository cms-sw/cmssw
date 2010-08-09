$(function() {
    // When the testform is submitted…
    $("#testform").submit(function() {
        // post the form values via AJAX…
        var postdata = {name: $("#name").val()} ;
        $.post('/submit', postdata, function(data) {
            // and set the title with the result
            $("#title").html(data['title']) ;
	  });
        return false ;
    });
    $("#secondtestform").submit(function() {
        // post the form values via AJAX…
        var postdata = {name: $("#firstName").val(), surname: $("#familyName").val()} ;
        $.post('/submit', postdata, function(data) {
            // and set the title with the result
            $("#title").html(data['title']) ;
	  });
        return false ;
        });
    });

function getUrlVars()
{
    var vars = [], hash;
    var hashes = window.location.href.slice(window.location.href.indexOf('?') + 1).split('&');
    for(var i = 0; i < hashes.length; i++)
    {
        hash = hashes[i].split('=');
        vars.push(hash[0]);
        vars[hash[0]] = hash[1];
    }
    return vars;
}