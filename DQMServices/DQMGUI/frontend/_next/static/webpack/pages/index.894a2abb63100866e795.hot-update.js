webpackHotUpdate_N_E("pages/index",{

/***/ "./hooks/useUpdateInLiveMode.tsx":
/*!***************************************!*\
  !*** ./hooks/useUpdateInLiveMode.tsx ***!
  \***************************************/
/*! exports provided: useUpdateLiveMode */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "useUpdateLiveMode", function() { return useUpdateLiveMode; });
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_3__);


var _s = $RefreshSig$();




var useUpdateLiveMode = function useUpdateLiveMode() {
  _s();

  var current_time = new Date().getTime();

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_1__["useState"](current_time),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState, 2),
      not_older_than = _React$useState2[0],
      set_not_older_than = _React$useState2[1];

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_1__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_2__["store"]),
      set_updated_by_not_older_than = _React$useContext.set_updated_by_not_older_than,
      update = _React$useContext.update,
      set_update = _React$useContext.set_update;

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"])();
  var query = router.query;

  var create_an_interval = function create_an_interval() {
    var interval = setInterval(function () {
      set_not_older_than(function () {
        // 1 sec is 1000 milisec. we dividing by 10000 and multiply by 20, because we need to
        // have rounded sec. for exmaple: if it is 13, we need to have 20, or 36, we need to have 20 and etc.
        var seconds = Math.round(new Date().getTime() / 1000) + 20 * 1000;
        console.log(seconds, Math.round(new Date().getTime()));
        return seconds;
      });
    }, 20000);
    return interval;
  };

  react__WEBPACK_IMPORTED_MODULE_1__["useEffect"](function () {
    var interval = create_an_interval();

    if (!update) {
      clearInterval(interval);
    }
  }, [update, query.run_number, query.dataset_name, query.folder_path, query.search_dataset_name, query.search_run_number]);
  react__WEBPACK_IMPORTED_MODULE_1__["useEffect"](function () {
    if (true) {
      set_updated_by_not_older_than(not_older_than);
    }
  }, [not_older_than, update]);
  return {
    not_older_than: not_older_than,
    set_update: set_update,
    update: update
  };
};

_s(useUpdateLiveMode, "+QbV0XLCzp4bp5nVS1UFr3tcJ6g=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"]];
});

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vaG9va3MvdXNlVXBkYXRlSW5MaXZlTW9kZS50c3giXSwibmFtZXMiOlsidXNlVXBkYXRlTGl2ZU1vZGUiLCJjdXJyZW50X3RpbWUiLCJEYXRlIiwiZ2V0VGltZSIsIlJlYWN0Iiwibm90X29sZGVyX3RoYW4iLCJzZXRfbm90X29sZGVyX3RoYW4iLCJzdG9yZSIsInNldF91cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuIiwidXBkYXRlIiwic2V0X3VwZGF0ZSIsInJvdXRlciIsInVzZVJvdXRlciIsInF1ZXJ5IiwiY3JlYXRlX2FuX2ludGVydmFsIiwiaW50ZXJ2YWwiLCJzZXRJbnRlcnZhbCIsInNlY29uZHMiLCJNYXRoIiwicm91bmQiLCJjb25zb2xlIiwibG9nIiwiY2xlYXJJbnRlcnZhbCIsInJ1bl9udW1iZXIiLCJkYXRhc2V0X25hbWUiLCJmb2xkZXJfcGF0aCIsInNlYXJjaF9kYXRhc2V0X25hbWUiLCJzZWFyY2hfcnVuX251bWJlciJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBRUE7QUFDQTtBQUdPLElBQU1BLGlCQUFpQixHQUFHLFNBQXBCQSxpQkFBb0IsR0FBTTtBQUFBOztBQUNyQyxNQUFNQyxZQUFZLEdBQUcsSUFBSUMsSUFBSixHQUFXQyxPQUFYLEVBQXJCOztBQURxQyx3QkFHUUMsOENBQUEsQ0FBZUgsWUFBZixDQUhSO0FBQUE7QUFBQSxNQUc5QkksY0FIOEI7QUFBQSxNQUdkQyxrQkFIYzs7QUFBQSwwQkFRakNGLGdEQUFBLENBQWlCRywrREFBakIsQ0FSaUM7QUFBQSxNQUtuQ0MsNkJBTG1DLHFCQUtuQ0EsNkJBTG1DO0FBQUEsTUFNbkNDLE1BTm1DLHFCQU1uQ0EsTUFObUM7QUFBQSxNQU9uQ0MsVUFQbUMscUJBT25DQSxVQVBtQzs7QUFTckMsTUFBTUMsTUFBTSxHQUFHQyw2REFBUyxFQUF4QjtBQUNBLE1BQU1DLEtBQWlCLEdBQUdGLE1BQU0sQ0FBQ0UsS0FBakM7O0FBRUEsTUFBTUMsa0JBQWtCLEdBQUcsU0FBckJBLGtCQUFxQixHQUFNO0FBQy9CLFFBQU1DLFFBQVEsR0FBR0MsV0FBVyxDQUFDLFlBQU07QUFDakNWLHdCQUFrQixDQUFDLFlBQU07QUFDdkI7QUFDQTtBQUNBLFlBQU1XLE9BQU8sR0FBR0MsSUFBSSxDQUFDQyxLQUFMLENBQVcsSUFBSWpCLElBQUosR0FBV0MsT0FBWCxLQUF1QixJQUFsQyxJQUEwQyxLQUFLLElBQS9EO0FBQ0FpQixlQUFPLENBQUNDLEdBQVIsQ0FBWUosT0FBWixFQUFxQkMsSUFBSSxDQUFDQyxLQUFMLENBQVcsSUFBSWpCLElBQUosR0FBV0MsT0FBWCxFQUFYLENBQXJCO0FBQ0EsZUFBT2MsT0FBUDtBQUNELE9BTmlCLENBQWxCO0FBT0QsS0FSMkIsRUFRekIsS0FSeUIsQ0FBNUI7QUFTQSxXQUFPRixRQUFQO0FBQ0QsR0FYRDs7QUFhQVgsaURBQUEsQ0FBZ0IsWUFBTTtBQUNwQixRQUFNVyxRQUFRLEdBQUdELGtCQUFrQixFQUFuQzs7QUFDQSxRQUFJLENBQUNMLE1BQUwsRUFBYTtBQUNYYSxtQkFBYSxDQUFDUCxRQUFELENBQWI7QUFDRDtBQUNGLEdBTEQsRUFLRyxDQUNETixNQURDLEVBRURJLEtBQUssQ0FBQ1UsVUFGTCxFQUdEVixLQUFLLENBQUNXLFlBSEwsRUFJRFgsS0FBSyxDQUFDWSxXQUpMLEVBS0RaLEtBQUssQ0FBQ2EsbUJBTEwsRUFNRGIsS0FBSyxDQUFDYyxpQkFOTCxDQUxIO0FBY0F2QixpREFBQSxDQUFnQixZQUFNO0FBQ3BCLFFBQUksSUFBSixFQUFVO0FBQ1JJLG1DQUE2QixDQUFDSCxjQUFELENBQTdCO0FBQ0Q7QUFDRixHQUpELEVBSUcsQ0FBQ0EsY0FBRCxFQUFpQkksTUFBakIsQ0FKSDtBQU1BLFNBQU87QUFBRUosa0JBQWMsRUFBZEEsY0FBRjtBQUFrQkssY0FBVSxFQUFWQSxVQUFsQjtBQUE4QkQsVUFBTSxFQUFOQTtBQUE5QixHQUFQO0FBQ0QsQ0E5Q007O0dBQU1ULGlCO1VBU0lZLHFEIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4Ljg5NGEyYWJiNjMxMDA4NjZlNzk1LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XG5cbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0JztcbmltcG9ydCB7IHVzZVJvdXRlciB9IGZyb20gJ25leHQvcm91dGVyJztcbmltcG9ydCB7IFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XG5cbmV4cG9ydCBjb25zdCB1c2VVcGRhdGVMaXZlTW9kZSA9ICgpID0+IHtcbiAgY29uc3QgY3VycmVudF90aW1lID0gbmV3IERhdGUoKS5nZXRUaW1lKCk7XG5cbiAgY29uc3QgW25vdF9vbGRlcl90aGFuLCBzZXRfbm90X29sZGVyX3RoYW5dID0gUmVhY3QudXNlU3RhdGUoY3VycmVudF90aW1lKTtcbiAgY29uc3Qge1xuICAgIHNldF91cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuLFxuICAgIHVwZGF0ZSxcbiAgICBzZXRfdXBkYXRlLFxuICB9ID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSk7XG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcblxuICBjb25zdCBjcmVhdGVfYW5faW50ZXJ2YWwgPSAoKSA9PiB7XG4gICAgY29uc3QgaW50ZXJ2YWwgPSBzZXRJbnRlcnZhbCgoKSA9PiB7XG4gICAgICBzZXRfbm90X29sZGVyX3RoYW4oKCkgPT4ge1xuICAgICAgICAvLyAxIHNlYyBpcyAxMDAwIG1pbGlzZWMuIHdlIGRpdmlkaW5nIGJ5IDEwMDAwIGFuZCBtdWx0aXBseSBieSAyMCwgYmVjYXVzZSB3ZSBuZWVkIHRvXG4gICAgICAgIC8vIGhhdmUgcm91bmRlZCBzZWMuIGZvciBleG1hcGxlOiBpZiBpdCBpcyAxMywgd2UgbmVlZCB0byBoYXZlIDIwLCBvciAzNiwgd2UgbmVlZCB0byBoYXZlIDIwIGFuZCBldGMuXG4gICAgICAgIGNvbnN0IHNlY29uZHMgPSBNYXRoLnJvdW5kKG5ldyBEYXRlKCkuZ2V0VGltZSgpIC8gMTAwMCkgKyAyMCAqIDEwMDA7XG4gICAgICAgIGNvbnNvbGUubG9nKHNlY29uZHMsIE1hdGgucm91bmQobmV3IERhdGUoKS5nZXRUaW1lKCkpKVxuICAgICAgICByZXR1cm4gc2Vjb25kcztcbiAgICAgIH0pO1xuICAgIH0sIDIwMDAwKTtcbiAgICByZXR1cm4gaW50ZXJ2YWw7XG4gIH07XG5cbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcbiAgICBjb25zdCBpbnRlcnZhbCA9IGNyZWF0ZV9hbl9pbnRlcnZhbCgpO1xuICAgIGlmICghdXBkYXRlKSB7XG4gICAgICBjbGVhckludGVydmFsKGludGVydmFsKTtcbiAgICB9XG4gIH0sIFtcbiAgICB1cGRhdGUsXG4gICAgcXVlcnkucnVuX251bWJlcixcbiAgICBxdWVyeS5kYXRhc2V0X25hbWUsXG4gICAgcXVlcnkuZm9sZGVyX3BhdGgsXG4gICAgcXVlcnkuc2VhcmNoX2RhdGFzZXRfbmFtZSxcbiAgICBxdWVyeS5zZWFyY2hfcnVuX251bWJlcixcbiAgXSk7XG5cbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcbiAgICBpZiAodHJ1ZSkge1xuICAgICAgc2V0X3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4obm90X29sZGVyX3RoYW4pO1xuICAgIH1cbiAgfSwgW25vdF9vbGRlcl90aGFuLCB1cGRhdGVdKTtcblxuICByZXR1cm4geyBub3Rfb2xkZXJfdGhhbiwgc2V0X3VwZGF0ZSwgdXBkYXRlIH07XG59O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==